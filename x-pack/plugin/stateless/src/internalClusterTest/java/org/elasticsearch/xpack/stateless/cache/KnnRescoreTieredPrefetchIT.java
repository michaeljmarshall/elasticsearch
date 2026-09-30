/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the Elastic License
 * 2.0; you may not use this file except in compliance with the Elastic License
 * 2.0.
 */

package org.elasticsearch.xpack.stateless.cache;

import org.apache.lucene.index.VectorSimilarityFunction;
import org.elasticsearch.action.bulk.BulkRequestBuilder;
import org.elasticsearch.action.search.SearchResponse;
import org.elasticsearch.blobcache.BlobCacheMetrics;
import org.elasticsearch.blobcache.shared.SharedBlobCacheService;
import org.elasticsearch.blobcache.shared.SharedBytes;
import org.elasticsearch.common.settings.Settings;
import org.elasticsearch.common.unit.ByteSizeValue;
import org.elasticsearch.index.IndexSettings;
import org.elasticsearch.license.License;
import org.elasticsearch.license.XPackLicenseState;
import org.elasticsearch.license.internal.XPackLicenseStatus;
import org.elasticsearch.plugins.Plugin;
import org.elasticsearch.search.SearchHit;
import org.elasticsearch.search.profile.SearchProfileDfsPhaseResult;
import org.elasticsearch.search.profile.SearchProfileShardResult;
import org.elasticsearch.search.profile.query.QueryProfileShardResult;
import org.elasticsearch.search.profile.query.TieredPrefetchOutcomeCounts;
import org.elasticsearch.search.vectors.KnnSearchBuilder;
import org.elasticsearch.search.vectors.RescoreVectorBuilder;
import org.elasticsearch.telemetry.Measurement;
import org.elasticsearch.telemetry.TestTelemetryPlugin;
import org.elasticsearch.xcontent.XContentBuilder;
import org.elasticsearch.xcontent.XContentFactory;
import org.elasticsearch.xpack.diskbbq.DiskBBQPlugin;
import org.elasticsearch.xpack.stateless.AbstractStatelessPluginIntegTestCase;
import org.elasticsearch.xpack.stateless.lucene.SearchDirectory;

import java.io.IOException;
import java.util.ArrayList;
import java.util.Collection;
import java.util.List;

import static org.elasticsearch.test.hamcrest.ElasticsearchAssertions.assertAcked;
import static org.elasticsearch.test.hamcrest.ElasticsearchAssertions.assertNoFailures;
import static org.elasticsearch.xpack.stateless.lucene.BlobStoreCacheDirectoryTestUtils.getCacheService;
import static org.hamcrest.Matchers.equalTo;
import static org.hamcrest.Matchers.greaterThan;
import static org.hamcrest.Matchers.notNullValue;

/**
 * End-to-end check of the tiered prefetch path for kNN rescoring on a stateless search node.
 *
 * <p>On a search node, vector data is read through {@code BlobCacheIndexInput}, which implements
 * {@link org.elasticsearch.core.TieredPrefetchInput}. When a top-level {@code knn} search with {@code rescore_vector} runs,
 * {@code RescoreKnnVectorQuery} asks the input to make every rescore candidate's raw vector resident in one bulk call per
 * leaf and tallies the outcomes, which are reported as {@code tiered_prefetch} under {@code profile.shards[].dfs.knn[]}.
 *
 * <p>Each scenario indexes a few hundred 64-dimensional float vectors on the indexing node, flushes them to the object
 * store, makes them searchable on the search node and evicts the search node's blob cache, so the first search runs
 * against a cold cache. The cold search must report non-resident outcomes and schedule object store fetches, which are
 * also visible on the {@code es.blob_cache.prefetch.total} metric. A second, identical search must then find every
 * candidate resident. In all runs, the returned scores must be the exact float similarities, proving that the prefetch
 * outcomes never change the results.
 */
public class KnnRescoreTieredPrefetchIT extends AbstractStatelessPluginIntegTestCase {

    private static final String VECTOR_FIELD = "vector";
    private static final int DIMS = 64;
    private static final int K = 10;
    private static final int NUM_CANDIDATES = 50;
    private static final float OVERSAMPLE = 2.0f;
    /** The number of candidates the rescore phase considers per shard: {@code ceil(k * oversample)}. */
    private static final int RESCORE_K = (int) Math.ceil(K * OVERSAMPLE);
    /**
     * Small regions so that the raw vector data of a few hundred documents spans many regions, and a cold search has to
     * fetch several of them.
     */
    private static final int REGION_SIZE_IN_BYTES = 4 * SharedBytes.PAGE_SIZE;

    @Override
    protected Collection<Class<? extends Plugin>> nodePlugins() {
        var plugins = new ArrayList<>(super.nodePlugins());
        plugins.add(TestTelemetryPlugin.class);
        plugins.add(DiskBBQPluginWithTrialLicense.class);
        return plugins;
    }

    /**
     * {@code bbq_disk} is provided by {@link DiskBBQPlugin} and requires an enterprise license, which it reads from the
     * shared x-pack license state. Stateless integration tests do not install x-pack core's license service, so grant a
     * trial license directly, as {@code TestUtils.StatelessPluginWithTrialLicense} does for the stateless plugin itself.
     */
    public static class DiskBBQPluginWithTrialLicense extends DiskBBQPlugin {
        public DiskBBQPluginWithTrialLicense(Settings settings) {
            super(settings);
        }

        @Override
        protected XPackLicenseState getLicenseState() {
            return new XPackLicenseState(System::currentTimeMillis, new XPackLicenseStatus(License.OperationMode.TRIAL, true, null));
        }
    }

    @Override
    protected Settings.Builder nodeSettings() {
        return super.nodeSettings()
            // Nothing but the searches under test may populate the search node's cache: no prefetching of new commits
            // on refresh notifications, and no cache warming when the search shard recovers.
            .put(SearchCommitPrefetcherDynamicSettings.PREFETCH_COMMITS_UPON_NOTIFICATIONS_ENABLED_SETTING.getKey(), false)
            .put(SharedBlobCacheWarmingService.SEARCH_OFFLINE_WARMING_ENABLED_SETTING.getKey(), false)
            .put(StatelessOnlinePrewarmingService.STATELESS_ONLINE_PREWARMING_ENABLED.getKey(), false);
    }

    /**
     * Fixed cache geometry, overriding the random one the base class picks: many small regions and a cache large enough
     * that nothing read by the first search is evicted before the second.
     */
    private static Settings.Builder cacheSettings() {
        return Settings.builder()
            .put(SharedBlobCacheService.SHARED_CACHE_REGION_SIZE_SETTING.getKey(), ByteSizeValue.ofBytes(REGION_SIZE_IN_BYTES))
            .put(SharedBlobCacheService.SHARED_CACHE_RANGE_SIZE_SETTING.getKey(), ByteSizeValue.ofBytes(REGION_SIZE_IN_BYTES))
            .put(
                SharedBlobCacheService.SHARED_CACHE_SIZE_SETTING.getKey(),
                ByteSizeValue.ofBytes(1024L * REGION_SIZE_IN_BYTES).getStringRep()
            );
    }

    /**
     * HNSW with BBQ quantization: the inner query already returns {@code rescoreK} hits, so the rescore is inline and
     * considers exactly {@code rescoreK} candidates.
     */
    public void testColdThenWarmRescoreBbqHnsw() throws Exception {
        runColdThenWarm("bbq_hnsw");
    }

    /**
     * DiskBBQ (IVF): exercises the posting list prefetching in {@code PrefetchingCentroidIterator} during the approximate
     * search, followed by a late rescore of the inner query's top {@code rescoreK} hits.
     */
    public void testColdThenWarmRescoreBbqDisk() throws Exception {
        runColdThenWarm("bbq_disk");
    }

    private void runColdThenWarm(String indexType) throws Exception {
        startMasterOnlyNode();
        startIndexNode(cacheSettings().build());
        String searchNode = startSearchNode(cacheSettings().build());
        ensureStableCluster(3);

        String indexName = randomIdentifier();
        float[][] vectors = createAndPopulateIndex(indexName, indexType);
        evictSearchNodeCache(indexName);

        float[] queryVector = randomVector();
        TestTelemetryPlugin telemetry = getTelemetryPlugin(searchNode);
        telemetry.resetMeter();

        // Cold run: the candidates' raw vectors are not in the search node's cache.
        TieredPrefetchOutcomeCounts cold = searchAndGetOutcomes(searchNode, indexName, queryVector, vectors);
        logger.info("--> cold run outcomes for [{}]: {}", indexType, cold);
        assertThat("each rescore candidate reports exactly one outcome", cold.total(), equalTo((long) RESCORE_K));
        assertThat("a cold cache cannot have every candidate resident", cold.fetching() + cold.skipped(), greaterThan(0L));

        // The FETCHING outcomes started asynchronous object store fetches that complete with a Fetched measurement. With
        // a budget of 32 regions and a cold cache, at least one region of a candidate must have been admitted.
        assertBusy(() -> {
            long fetched = prefetchMeasurements(telemetry, BlobCacheMetrics.PrefetchResult.Fetched);
            long skipped = prefetchMeasurements(telemetry, BlobCacheMetrics.PrefetchResult.Skipped);
            assertThat("the cold search should have scheduled object store prefetches", fetched + skipped, greaterThan(0L));
        });
        if (cold.fetching() > 0) {
            assertBusy(() -> assertThat(prefetchMeasurements(telemetry, BlobCacheMetrics.PrefetchResult.Fetched), greaterThan(0L)));
        }
        // Wait for every admitted fetch to finish, so that the warm run does not join a fetch still in flight.
        SearchDirectory searchDirectory = SearchDirectory.unwrapDirectory(findSearchShard(indexName).store().directory());
        assertBusy(() -> assertThat(getCacheService(searchDirectory).getPrefetchBudget().inFlightRegions(), equalTo(0)));

        // Warm run: the same query picks the same candidates, whose vectors the cold run read into the cache.
        TieredPrefetchOutcomeCounts warm = searchAndGetOutcomes(searchNode, indexName, queryVector, vectors);
        logger.info("--> warm run outcomes for [{}]: {}", indexType, warm);
        assertThat(warm.total(), equalTo((long) RESCORE_K));
        assertThat("nothing should be fetched once the candidates are cached", warm.fetching(), equalTo(0L));
        assertThat(warm.skipped(), equalTo(0L));
        assertThat(warm.resident(), equalTo((long) RESCORE_K));
    }

    /**
     * With a prefetch budget of zero regions the search node never admits an {@code ensureResident} fetch, so every
     * non-resident candidate is reported as skipped, and the {@code Skipped} metric (which only {@code ensureResident}
     * records) is incremented. The rescore then reads the skipped vectors through the blocking read path and still returns
     * exact scores.
     */
    public void testZeroPrefetchBudgetSkipsButStillRescoresCorrectly() throws Exception {
        startMasterOnlyNode();
        startIndexNode(cacheSettings().build());
        String searchNode = startSearchNode(
            cacheSettings().put(
                StatelessSharedBlobCacheService.STATELESS_CACHE_OBJECT_STORE_PREFETCH_MAX_IN_FLIGHT_REGIONS_SETTING.getKey(),
                0
            ).build()
        );
        ensureStableCluster(3);

        String indexName = randomIdentifier();
        float[][] vectors = createAndPopulateIndex(indexName, "bbq_hnsw");
        evictSearchNodeCache(indexName);

        float[] queryVector = randomVector();
        TestTelemetryPlugin telemetry = getTelemetryPlugin(searchNode);
        telemetry.resetMeter();

        TieredPrefetchOutcomeCounts cold = searchAndGetOutcomes(searchNode, indexName, queryVector, vectors);
        logger.info("--> cold run outcomes with a zero prefetch budget: {}", cold);
        assertThat(cold.total(), equalTo((long) RESCORE_K));
        assertThat("no fetch can be admitted with a zero budget", cold.fetching(), equalTo(0L));
        assertThat("a cold cache cannot have every candidate resident", cold.skipped(), greaterThan(0L));
        assertThat(prefetchMeasurements(telemetry, BlobCacheMetrics.PrefetchResult.Skipped), greaterThan(0L));

        // The blocking reads of the cold run populated the cache, so a rerun finds every candidate resident.
        TieredPrefetchOutcomeCounts warm = searchAndGetOutcomes(searchNode, indexName, queryVector, vectors);
        logger.info("--> warm run outcomes with a zero prefetch budget: {}", warm);
        assertThat(warm.resident(), equalTo((long) RESCORE_K));
    }

    /**
     * Creates a single-shard index with one search replica, indexes random vectors in one bulk, flushes so the segment
     * is uploaded to the object store, and refreshes so the search node opens it. Returns the vectors by document id.
     */
    private float[][] createAndPopulateIndex(String indexName, String indexType) throws IOException {
        XContentBuilder mapping = XContentFactory.jsonBuilder()
            .startObject()
            .startObject("properties")
            .startObject(VECTOR_FIELD)
            .field("type", "dense_vector")
            .field("dims", DIMS)
            .field("index", true)
            .field("similarity", "l2_norm")
            .startObject("index_options")
            .field("type", indexType)
            .endObject()
            .endObject()
            .endObject()
            .endObject();
        assertAcked(
            prepareCreate(indexName).setSettings(indexSettings(1, 1).put(IndexSettings.INDEX_REFRESH_INTERVAL_SETTING.getKey(), -1))
                .setMapping(mapping)
        );
        ensureGreen(indexName);

        int numDocs = randomIntBetween(400, 600);
        float[][] vectors = new float[numDocs][];
        BulkRequestBuilder bulk = client().prepareBulk();
        for (int i = 0; i < numDocs; i++) {
            vectors[i] = randomVector();
            bulk.add(client().prepareIndex(indexName).setId(Integer.toString(i)).setSource(VECTOR_FIELD, vectors[i]));
        }
        assertNoFailures(bulk.get());
        flush(indexName);
        refresh(indexName);
        return vectors;
    }

    /**
     * Drops everything from the search node's blob cache. The refresh has already opened the new segment on the search
     * node, so this leaves the shard searchable with a cold cache.
     */
    private static void evictSearchNodeCache(String indexName) {
        SearchDirectory searchDirectory = SearchDirectory.unwrapDirectory(findSearchShard(indexName).store().directory());
        getCacheService(searchDirectory).forceEvict(key -> true);
    }

    /**
     * Runs a top-level kNN search with rescoring and profiling through the search node, checks that every hit's score is
     * the exact float similarity of its vector, and returns the tiered prefetch outcomes summed over the shards' kNN
     * profiles.
     */
    private TieredPrefetchOutcomeCounts searchAndGetOutcomes(String node, String indexName, float[] queryVector, float[][] vectors) {
        var knn = new KnnSearchBuilder(VECTOR_FIELD, queryVector, K, NUM_CANDIDATES, null, new RescoreVectorBuilder(OVERSAMPLE), null);
        SearchResponse response = client(node).prepareSearch(indexName).setKnnSearch(List.of(knn)).setSize(K).setProfile(true).get();
        try {
            assertNoFailures(response);
            assertThat(response.getHits().getHits().length, equalTo(K));
            for (SearchHit hit : response.getHits().getHits()) {
                float expected = VectorSimilarityFunction.EUCLIDEAN.compare(queryVector, vectors[Integer.parseInt(hit.getId())]);
                assertEquals("rescored score of doc [" + hit.getId() + "]", expected, hit.getScore(), 1e-5f);
            }

            TieredPrefetchOutcomeCounts total = new TieredPrefetchOutcomeCounts();
            int shardsWithOutcomes = 0;
            for (SearchProfileShardResult shard : response.getSearchProfileShardResults().values()) {
                SearchProfileDfsPhaseResult dfs = shard.getSearchProfileDfsPhaseResult();
                assertThat("a top-level knn search runs in the dfs phase", dfs, notNullValue());
                assertThat(dfs.getQueryProfileShardResult(), notNullValue());
                for (QueryProfileShardResult knnProfile : dfs.getQueryProfileShardResult()) {
                    TieredPrefetchOutcomeCounts outcomes = knnProfile.getTieredPrefetchOutcomes();
                    if (outcomes != null) {
                        shardsWithOutcomes++;
                        total.add(outcomes);
                    }
                }
            }
            assertThat("the search node reads vectors through a tiered input", shardsWithOutcomes, greaterThan(0));
            return total;
        } finally {
            response.decRef();
        }
    }

    private static long prefetchMeasurements(TestTelemetryPlugin telemetry, BlobCacheMetrics.PrefetchResult result) {
        return telemetry.getLongCounterMeasurement(BlobCacheMetrics.BLOB_CACHE_PREFETCH_TOTAL)
            .stream()
            .filter(m -> result.name().equals(m.attributes().get(BlobCacheMetrics.PREFETCH_RESULT_ATTRIBUTE_KEY)))
            .mapToLong(Measurement::getLong)
            .sum();
    }

    private static float[] randomVector() {
        float[] vector = new float[DIMS];
        for (int i = 0; i < DIMS; i++) {
            vector[i] = randomFloatBetween(-1f, 1f, true);
        }
        return vector;
    }
}
