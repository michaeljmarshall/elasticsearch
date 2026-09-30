/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the Elastic License
 * 2.0; you may not use this file except in compliance with the Elastic License
 * 2.0.
 */

package org.elasticsearch.xpack.stateless.cache.reader;

import org.apache.lucene.store.IOContext;
import org.elasticsearch.ResourceAlreadyUploadedException;
import org.elasticsearch.action.ActionListener;
import org.elasticsearch.action.support.PlainActionFuture;
import org.elasticsearch.blobcache.BlobCacheMetrics;
import org.elasticsearch.blobcache.CachePopulationSource;
import org.elasticsearch.blobcache.shared.SharedBlobCacheService;
import org.elasticsearch.blobcache.shared.SharedBytes;
import org.elasticsearch.common.settings.Settings;
import org.elasticsearch.common.unit.ByteSizeValue;
import org.elasticsearch.common.util.concurrent.EsExecutors;
import org.elasticsearch.core.TieredPrefetchInput.Outcome;
import org.elasticsearch.env.Environment;
import org.elasticsearch.env.NodeEnvironment;
import org.elasticsearch.env.TestEnvironment;
import org.elasticsearch.index.Index;
import org.elasticsearch.index.shard.ShardId;
import org.elasticsearch.telemetry.InstrumentType;
import org.elasticsearch.telemetry.Measurement;
import org.elasticsearch.telemetry.RecordingMeterRegistry;
import org.elasticsearch.test.ESTestCase;
import org.elasticsearch.threadpool.TestThreadPool;
import org.elasticsearch.threadpool.ThreadPool;
import org.elasticsearch.xpack.searchablesnapshots.cache.common.TestUtils;
import org.elasticsearch.xpack.stateless.StatelessPlugin;
import org.elasticsearch.xpack.stateless.cache.StatelessSharedBlobCacheService;
import org.elasticsearch.xpack.stateless.lucene.FileCacheKey;
import org.junit.After;
import org.junit.Before;

import java.io.InputStream;
import java.nio.ByteBuffer;
import java.util.List;
import java.util.concurrent.CountDownLatch;
import java.util.concurrent.atomic.AtomicInteger;

import static org.elasticsearch.blobcache.shared.SharedBlobCacheServiceTestUtils.randomRegionTimestampMillis;
import static org.elasticsearch.xpack.stateless.TestUtils.NOOP_TIME_PROVIDER;
import static org.elasticsearch.xpack.stateless.TestUtils.newCacheService;
import static org.elasticsearch.xpack.stateless.commits.BlobLocationTestUtils.createBlobFileRanges;
import static org.hamcrest.Matchers.equalTo;
import static org.hamcrest.Matchers.greaterThan;

public class CacheFileReaderTests extends ESTestCase {

    private static final int REGION_PAGES = 10;
    private static final int BLOB_LENGTH = REGION_PAGES * SharedBytes.PAGE_SIZE;
    private static final ByteSizeValue REGION_SIZE = ByteSizeValue.ofBytes(BLOB_LENGTH);

    private ThreadPool threadPool;

    @Before
    public void startThreadPool() throws Exception {
        threadPool = new TestThreadPool("CacheFileReaderTests", StatelessPlugin.statelessExecutorBuilders(Settings.EMPTY, randomBoolean()));
    }

    @After
    public void stopThreadPool() throws Exception {
        assertTrue(terminate(threadPool));
    }

    public void testTryPrefetchFetches() throws Exception {
        Settings settings = nodeSettings();
        RecordingMeterRegistry meterRegistry = new RecordingMeterRegistry();
        BlobCacheMetrics metrics = new BlobCacheMetrics(meterRegistry, NOOP_TIME_PROVIDER);

        try (
            NodeEnvironment env = new NodeEnvironment(settings, TestEnvironment.newEnvironment(settings));
            StatelessSharedBlobCacheService service = newCacheService(env, settings, threadPool)
        ) {
            String fileName = "prefetch-target";
            byte[] blob = randomByteArrayOfLength(BLOB_LENGTH);
            FileCacheKey cacheKey = new FileCacheKey(new ShardId(new Index("idx", "uid"), 0), 1L, fileName);
            AtomicInteger fetchCount = new AtomicInteger();
            CacheBlobReader reader = countingObjectStoreReader(fileName, blob, service.getRangeSize(), fetchCount);
            CacheFileReader cacheFileReader = new CacheFileReader(
                service.getCacheFile(cacheKey, blob.length, SharedBlobCacheService.CacheMissHandler.NOOP, randomRegionTimestampMillis()),
                reader,
                createBlobFileRanges(1L, 0L, 0, blob.length),
                metrics,
                System::currentTimeMillis,
                true
            );

            assertFalse("first call should miss the fast path", cacheFileReader.tryPrefetch(0L, blob.length));
            assertThat("blob store should have served at least one range request", fetchCount.get(), greaterThan(0));
            assertPrefetchMetric(meterRegistry, BlobCacheMetrics.PrefetchResult.Fetched, 1);
            assertPrefetchMetric(meterRegistry, BlobCacheMetrics.PrefetchResult.Failed, 0);
            assertPrefetchMetric(meterRegistry, BlobCacheMetrics.PrefetchResult.AlreadyCached, 0);

            int fetchCountAfterPopulate = fetchCount.get();
            assertTrue("second call should hit the fast path now that the range is cached", cacheFileReader.tryPrefetch(0L, blob.length));
            assertThat("fast path must not re-fetch from the blob store", fetchCount.get(), equalTo(fetchCountAfterPopulate));
            assertPrefetchMetric(meterRegistry, BlobCacheMetrics.PrefetchResult.AlreadyCached, 1);
        }
    }

    public void testTryPrefetchRecordsFailure() throws Exception {
        Settings settings = nodeSettings();
        RecordingMeterRegistry meterRegistry = new RecordingMeterRegistry();
        BlobCacheMetrics metrics = new BlobCacheMetrics(meterRegistry, NOOP_TIME_PROVIDER);

        try (
            NodeEnvironment env = new NodeEnvironment(settings, TestEnvironment.newEnvironment(settings));
            StatelessSharedBlobCacheService service = newCacheService(env, settings, threadPool)
        ) {
            String fileName = "prefetch-failure";
            byte[] blob = randomByteArrayOfLength(BLOB_LENGTH);
            FileCacheKey cacheKey = new FileCacheKey(new ShardId(new Index("idx", "uid"), 0), 1L, fileName);
            CacheBlobReader reader = new ObjectStoreCacheBlobReader(
                TestUtils.singleBlobContainer(fileName, blob),
                fileName,
                service.getRangeSize(),
                EsExecutors.DIRECT_EXECUTOR_SERVICE
            ) {
                @Override
                public void getRangeInputStream(long position, int length, ActionListener<InputStream> listener) {
                    listener.onFailure(new java.io.IOException("simulated blob fetch failure"));
                }
            };
            CacheFileReader cacheFileReader = new CacheFileReader(
                service.getCacheFile(cacheKey, blob.length, SharedBlobCacheService.CacheMissHandler.NOOP, randomRegionTimestampMillis()),
                reader,
                createBlobFileRanges(1L, 0L, 0, blob.length),
                metrics,
                System::currentTimeMillis,
                true
            );

            assertFalse(cacheFileReader.tryPrefetch(0L, blob.length));
            assertPrefetchMetric(meterRegistry, BlobCacheMetrics.PrefetchResult.Failed, 1);
            assertPrefetchMetric(meterRegistry, BlobCacheMetrics.PrefetchResult.Fetched, 0);
            assertPrefetchMetric(meterRegistry, BlobCacheMetrics.PrefetchResult.AlreadyCached, 0);
        }
    }

    public void testTryPrefetchWithOversizedFileLength() throws Exception {
        Settings settings = nodeSettings();
        RecordingMeterRegistry meterRegistry = new RecordingMeterRegistry();
        BlobCacheMetrics metrics = new BlobCacheMetrics(meterRegistry, NOOP_TIME_PROVIDER);

        try (
            NodeEnvironment env = new NodeEnvironment(settings, TestEnvironment.newEnvironment(settings));
            StatelessSharedBlobCacheService service = newCacheService(env, settings, threadPool)
        ) {
            String fileName = "prefetch-oversized-length";
            byte[] blob = randomByteArrayOfLength(BLOB_LENGTH);
            FileCacheKey cacheKey = new FileCacheKey(new ShardId(new Index("idx", "uid"), 0), 1L, fileName);
            AtomicInteger fetchCount = new AtomicInteger();
            CacheBlobReader reader = countingObjectStoreReader(fileName, blob, service.getRangeSize(), fetchCount);
            CacheFileReader cacheFileReader = new CacheFileReader(
                service.getCacheFile(cacheKey, blob.length, SharedBlobCacheService.CacheMissHandler.NOOP, randomRegionTimestampMillis()),
                reader,
                createBlobFileRanges(1L, 0L, 0, blob.length),
                metrics,
                System::currentTimeMillis,
                true
            );

            long oversizedLength = (long) blob.length * 1024L;
            assertFalse(
                "oversized prefetch must not hit the fast path on the first call",
                cacheFileReader.tryPrefetch(0L, oversizedLength)
            );
            assertThat("oversized prefetch must still trigger a fetch from the blob store", fetchCount.get(), greaterThan(0));
            assertPrefetchMetric(meterRegistry, BlobCacheMetrics.PrefetchResult.Fetched, 1);
            assertPrefetchMetric(meterRegistry, BlobCacheMetrics.PrefetchResult.Failed, 0);
            assertPrefetchMetric(meterRegistry, BlobCacheMetrics.PrefetchResult.AlreadyCached, 0);
        }
    }

    public void testTryPrefetchPastEOF() throws Exception {
        Settings settings = nodeSettings();
        RecordingMeterRegistry meterRegistry = new RecordingMeterRegistry();
        BlobCacheMetrics metrics = new BlobCacheMetrics(meterRegistry, NOOP_TIME_PROVIDER);

        try (
            NodeEnvironment env = new NodeEnvironment(settings, TestEnvironment.newEnvironment(settings));
            StatelessSharedBlobCacheService service = newCacheService(env, settings, threadPool)
        ) {
            String fileName = "prefetch-past-eof";
            byte[] blob = randomByteArrayOfLength(BLOB_LENGTH);
            FileCacheKey cacheKey = new FileCacheKey(new ShardId(new Index("idx", "uid"), 0), 1L, fileName);
            AtomicInteger fetchCount = new AtomicInteger();
            CacheBlobReader reader = countingObjectStoreReader(fileName, blob, service.getRangeSize(), fetchCount);
            CacheFileReader cacheFileReader = new CacheFileReader(
                service.getCacheFile(cacheKey, blob.length, SharedBlobCacheService.CacheMissHandler.NOOP, randomRegionTimestampMillis()),
                reader,
                createBlobFileRanges(1L, 0L, 0, blob.length),
                metrics,
                System::currentTimeMillis,
                true
            );

            long offsetAtOrPastEof = randomBoolean() ? blob.length : blob.length + randomLongBetween(1L, 1024L);
            assertFalse(cacheFileReader.tryPrefetch(offsetAtOrPastEof, randomLongBetween(1L, 1024L)));
            assertThat("no fetch should be triggered when the offset is past EOF", fetchCount.get(), equalTo(0));
            assertPrefetchMetric(meterRegistry, BlobCacheMetrics.PrefetchResult.AlreadyCached, 0);
            assertPrefetchMetric(meterRegistry, BlobCacheMetrics.PrefetchResult.Fetched, 0);
            assertPrefetchMetric(meterRegistry, BlobCacheMetrics.PrefetchResult.Failed, 0);
        }
    }

    public void testTryPrefetchNonPositiveLength() throws Exception {
        Settings settings = nodeSettings();
        RecordingMeterRegistry meterRegistry = new RecordingMeterRegistry();
        BlobCacheMetrics metrics = new BlobCacheMetrics(meterRegistry, NOOP_TIME_PROVIDER);

        try (
            NodeEnvironment env = new NodeEnvironment(settings, TestEnvironment.newEnvironment(settings));
            StatelessSharedBlobCacheService service = newCacheService(env, settings, threadPool)
        ) {
            String fileName = "prefetch-zero-length";
            byte[] blob = randomByteArrayOfLength(BLOB_LENGTH);
            FileCacheKey cacheKey = new FileCacheKey(new ShardId(new Index("idx", "uid"), 0), 1L, fileName);
            AtomicInteger fetchCount = new AtomicInteger();
            CacheBlobReader reader = countingObjectStoreReader(fileName, blob, service.getRangeSize(), fetchCount);
            CacheFileReader cacheFileReader = new CacheFileReader(
                service.getCacheFile(cacheKey, blob.length, SharedBlobCacheService.CacheMissHandler.NOOP, randomRegionTimestampMillis()),
                reader,
                createBlobFileRanges(1L, 0L, 0, blob.length),
                metrics,
                System::currentTimeMillis,
                true
            );

            long nonPositiveLength = randomBoolean() ? 0L : -randomLongBetween(1L, 1024L);
            assertFalse(cacheFileReader.tryPrefetch(0L, nonPositiveLength));
            assertThat("no fetch should be triggered when length is non-positive", fetchCount.get(), equalTo(0));
            assertPrefetchMetric(meterRegistry, BlobCacheMetrics.PrefetchResult.AlreadyCached, 0);
            assertPrefetchMetric(meterRegistry, BlobCacheMetrics.PrefetchResult.Fetched, 0);
            assertPrefetchMetric(meterRegistry, BlobCacheMetrics.PrefetchResult.Failed, 0);
        }
    }

    public void testTryPrefetchOversizedLengthIsLimited() throws Exception {
        Settings settings = nodeSettings();
        RecordingMeterRegistry meterRegistry = new RecordingMeterRegistry();
        BlobCacheMetrics metrics = new BlobCacheMetrics(meterRegistry, NOOP_TIME_PROVIDER);

        try (
            NodeEnvironment env = new NodeEnvironment(settings, TestEnvironment.newEnvironment(settings));
            StatelessSharedBlobCacheService service = newCacheService(env, settings, threadPool)
        ) {
            String fileName = "prefetch-midfile-oversized";
            byte[] blob = randomByteArrayOfLength(BLOB_LENGTH);
            FileCacheKey cacheKey = new FileCacheKey(new ShardId(new Index("idx", "uid"), 0), 1L, fileName);
            AtomicInteger fetchCount = new AtomicInteger();
            CacheBlobReader reader = countingObjectStoreReader(fileName, blob, service.getRangeSize(), fetchCount);
            CacheFileReader cacheFileReader = new CacheFileReader(
                service.getCacheFile(cacheKey, blob.length, SharedBlobCacheService.CacheMissHandler.NOOP, randomRegionTimestampMillis()),
                reader,
                createBlobFileRanges(1L, 0L, 0, blob.length),
                metrics,
                System::currentTimeMillis,
                true
            );

            long midFileOffset = randomLongBetween(1L, blob.length - 1);
            long oversizedLength = (blob.length - midFileOffset) + randomLongBetween(1L, blob.length);
            assertFalse(cacheFileReader.tryPrefetch(midFileOffset, oversizedLength));
            assertThat(fetchCount.get(), greaterThan(0));
            assertPrefetchMetric(meterRegistry, BlobCacheMetrics.PrefetchResult.Fetched, 1);
            assertPrefetchMetric(meterRegistry, BlobCacheMetrics.PrefetchResult.Failed, 0);
            assertPrefetchMetric(meterRegistry, BlobCacheMetrics.PrefetchResult.AlreadyCached, 0);
        }
    }

    public void testTryPrefetchRetriesOnAlreadyUploaded() throws Exception {
        Settings settings = nodeSettings();
        RecordingMeterRegistry meterRegistry = new RecordingMeterRegistry();
        BlobCacheMetrics metrics = new BlobCacheMetrics(meterRegistry, NOOP_TIME_PROVIDER);

        try (
            NodeEnvironment env = new NodeEnvironment(settings, TestEnvironment.newEnvironment(settings));
            StatelessSharedBlobCacheService service = newCacheService(env, settings, threadPool)
        ) {
            String fileName = "prefetch-already-uploaded-retry";
            byte[] blob = randomByteArrayOfLength(BLOB_LENGTH);
            FileCacheKey cacheKey = new FileCacheKey(new ShardId(new Index("idx", "uid"), 0), 1L, fileName);
            AtomicInteger fetchCount = new AtomicInteger();
            CacheBlobReader reader = alreadyUploadedThenServingReader(fileName, blob, service.getRangeSize(), 1, fetchCount);
            CacheFileReader cacheFileReader = new CacheFileReader(
                service.getCacheFile(cacheKey, blob.length, SharedBlobCacheService.CacheMissHandler.NOOP, randomRegionTimestampMillis()),
                reader,
                createBlobFileRanges(1L, 0L, 0, blob.length),
                metrics,
                System::currentTimeMillis,
                true
            );

            assertFalse(
                "first call should miss the fast path and schedule an async download",
                cacheFileReader.tryPrefetch(0L, blob.length)
            );
            assertThat("the fetch should have failed once and then succeeded on the retry", fetchCount.get(), equalTo(2));
            assertPrefetchMetric(meterRegistry, BlobCacheMetrics.PrefetchResult.Fetched, 1);
            assertPrefetchMetric(meterRegistry, BlobCacheMetrics.PrefetchResult.Failed, 0);
            assertPrefetchMetric(meterRegistry, BlobCacheMetrics.PrefetchResult.AlreadyCached, 0);

            assertTrue("the retry should have populated the cache", cacheFileReader.tryPrefetch(0L, blob.length));
            assertPrefetchMetric(meterRegistry, BlobCacheMetrics.PrefetchResult.AlreadyCached, 1);
            assertPrefetchMetric(meterRegistry, BlobCacheMetrics.PrefetchResult.Fetched, 1);
        }
    }

    public void testTryPrefetchFailsAfterMaxAlreadyUploadedRetries() throws Exception {
        Settings settings = nodeSettings();
        RecordingMeterRegistry meterRegistry = new RecordingMeterRegistry();
        BlobCacheMetrics metrics = new BlobCacheMetrics(meterRegistry, NOOP_TIME_PROVIDER);

        try (
            NodeEnvironment env = new NodeEnvironment(settings, TestEnvironment.newEnvironment(settings));
            StatelessSharedBlobCacheService service = newCacheService(env, settings, threadPool)
        ) {
            String fileName = "prefetch-already-uploaded-exhausted";
            byte[] blob = randomByteArrayOfLength(BLOB_LENGTH);
            FileCacheKey cacheKey = new FileCacheKey(new ShardId(new Index("idx", "uid"), 0), 1L, fileName);
            AtomicInteger fetchCount = new AtomicInteger();
            // Always fail with ResourceAlreadyUploadedException; if retries were unbounded this reader would be called forever.
            CacheBlobReader reader = alreadyUploadedThenServingReader(
                fileName,
                blob,
                service.getRangeSize(),
                Integer.MAX_VALUE,
                fetchCount
            );
            CacheFileReader cacheFileReader = new CacheFileReader(
                service.getCacheFile(cacheKey, blob.length, SharedBlobCacheService.CacheMissHandler.NOOP, randomRegionTimestampMillis()),
                reader,
                createBlobFileRanges(1L, 0L, 0, blob.length),
                metrics,
                System::currentTimeMillis,
                true
            );

            assertFalse(cacheFileReader.tryPrefetch(0L, blob.length));
            assertThat("prefetch must stop after exhausting the retry budget", fetchCount.get(), equalTo(3));
            assertPrefetchMetric(meterRegistry, BlobCacheMetrics.PrefetchResult.Failed, 1);
            assertPrefetchMetric(meterRegistry, BlobCacheMetrics.PrefetchResult.Fetched, 0);
            assertPrefetchMetric(meterRegistry, BlobCacheMetrics.PrefetchResult.AlreadyCached, 0);
        }
    }

    public void testTryPrefetchDoesNotRetryOnOtherError() throws Exception {
        Settings settings = nodeSettings();
        RecordingMeterRegistry meterRegistry = new RecordingMeterRegistry();
        BlobCacheMetrics metrics = new BlobCacheMetrics(meterRegistry, NOOP_TIME_PROVIDER);

        try (
            NodeEnvironment env = new NodeEnvironment(settings, TestEnvironment.newEnvironment(settings));
            StatelessSharedBlobCacheService service = newCacheService(env, settings, threadPool)
        ) {
            String fileName = "prefetch-other-error";
            byte[] blob = randomByteArrayOfLength(BLOB_LENGTH);
            FileCacheKey cacheKey = new FileCacheKey(new ShardId(new Index("idx", "uid"), 0), 1L, fileName);
            AtomicInteger fetchCount = new AtomicInteger();
            // A non-ResourceAlreadyUploadedException failure must not be retried.
            CacheBlobReader reader = new ObjectStoreCacheBlobReader(
                TestUtils.singleBlobContainer(fileName, blob),
                fileName,
                service.getRangeSize(),
                EsExecutors.DIRECT_EXECUTOR_SERVICE
            ) {
                @Override
                public void getRangeInputStream(long position, int length, ActionListener<InputStream> listener) {
                    fetchCount.incrementAndGet();
                    listener.onFailure(new java.io.IOException("simulated blob fetch failure"));
                }
            };
            CacheFileReader cacheFileReader = new CacheFileReader(
                service.getCacheFile(cacheKey, blob.length, SharedBlobCacheService.CacheMissHandler.NOOP, randomRegionTimestampMillis()),
                reader,
                createBlobFileRanges(1L, 0L, 0, blob.length),
                metrics,
                System::currentTimeMillis,
                true
            );

            assertFalse(cacheFileReader.tryPrefetch(0L, blob.length));
            assertThat(fetchCount.get(), equalTo(1));
            assertPrefetchMetric(meterRegistry, BlobCacheMetrics.PrefetchResult.Failed, 1);
            assertPrefetchMetric(meterRegistry, BlobCacheMetrics.PrefetchResult.Fetched, 0);
            assertPrefetchMetric(meterRegistry, BlobCacheMetrics.PrefetchResult.AlreadyCached, 0);
        }
    }

    public void testTryPrefetchDisabledOnlyUsesFastPath() throws Exception {
        Settings settings = nodeSettings();
        RecordingMeterRegistry meterRegistry = new RecordingMeterRegistry();
        BlobCacheMetrics metrics = new BlobCacheMetrics(meterRegistry, NOOP_TIME_PROVIDER);

        try (
            NodeEnvironment env = new NodeEnvironment(settings, TestEnvironment.newEnvironment(settings));
            StatelessSharedBlobCacheService service = newCacheService(env, settings, threadPool)
        ) {
            String fileName = "prefetch-disabled";
            byte[] blob = randomByteArrayOfLength(BLOB_LENGTH);
            FileCacheKey cacheKey = new FileCacheKey(new ShardId(new Index("idx", "uid"), 0), 1L, fileName);
            AtomicInteger fetchCount = new AtomicInteger();
            CacheBlobReader reader = countingObjectStoreReader(fileName, blob, service.getRangeSize(), fetchCount);
            CacheFileReader cacheFileReader = new CacheFileReader(
                service.getCacheFile(cacheKey, blob.length, SharedBlobCacheService.CacheMissHandler.NOOP, randomRegionTimestampMillis()),
                reader,
                createBlobFileRanges(1L, 0L, 0, blob.length),
                metrics,
                System::currentTimeMillis,
                false
            );

            assertFalse(
                "cache miss must not schedule an async download when object store prefetch is disabled",
                cacheFileReader.tryPrefetch(0L, blob.length)
            );
            assertThat("no fetch should be triggered when object store prefetch is disabled", fetchCount.get(), equalTo(0));
            assertPrefetchMetric(meterRegistry, BlobCacheMetrics.PrefetchResult.Fetched, 0);
            assertPrefetchMetric(meterRegistry, BlobCacheMetrics.PrefetchResult.Failed, 0);
            assertPrefetchMetric(meterRegistry, BlobCacheMetrics.PrefetchResult.AlreadyCached, 0);

            // populate the cache through a regular read; the fast path must still succeed when prefetch is disabled
            cacheFileReader.read(this, ByteBuffer.allocate(blob.length), 0, blob.length, blob.length, "test-plain");
            assertTrue(
                "fast path must still succeed for cached data when object store prefetch is disabled",
                cacheFileReader.tryPrefetch(0L, blob.length)
            );
        }
    }

    /**
     * {@code SEARCH_ORIGIN_REMOTE_STORAGE_DOWNLOAD_TOOK_TIME} must carry {@link CachePopulationSource#BlobStore}
     * for SEARCH-thread reads, {@link CachePopulationSource#Peer} for VBCC-thread reads, and no measurement
     * for non-{@code EsThread} callers.
     */
    public void testReadRecordsSearchOriginMetricWithCorrectAttributes() throws Exception {
        Settings settings = nodeSettings();
        RecordingMeterRegistry meterRegistry = new RecordingMeterRegistry();
        BlobCacheMetrics metrics = new BlobCacheMetrics(meterRegistry, NOOP_TIME_PROVIDER);

        try (
            NodeEnvironment env = new NodeEnvironment(settings, TestEnvironment.newEnvironment(settings));
            StatelessSharedBlobCacheService service = newCacheService(env, settings, threadPool)
        ) {
            String fileName = "read-attribute-test";
            byte[] blob = randomByteArrayOfLength(BLOB_LENGTH);
            FileCacheKey cacheKey = new FileCacheKey(new ShardId(new Index("idx", "uid"), 0), 1L, fileName);
            CacheBlobReader blobReader = countingObjectStoreReader(fileName, blob, service.getRangeSize(), new AtomicInteger());
            CacheFileReader cacheFileReader = new CacheFileReader(
                service.getCacheFile(cacheKey, blob.length, SharedBlobCacheService.CacheMissHandler.NOOP, randomRegionTimestampMillis()),
                blobReader,
                createBlobFileRanges(1L, 0L, 0, blob.length),
                metrics,
                System::currentTimeMillis,
                true
            );

            // Pre-populate the cache so read() calls go through the fast path
            // and do not block on SHARD_READ_THREAD_POOL
            assertFalse("first prefetch should schedule an async download", cacheFileReader.tryPrefetch(0L, blob.length));
            assertBusy(() -> assertTrue("cache should be fully populated", cacheFileReader.tryPrefetch(0L, blob.length)));
            meterRegistry.getRecorder().resetCalls();

            // SEARCH thread → BlobStore attribute
            PlainActionFuture<Void> searchFuture = new PlainActionFuture<>();
            threadPool.executor(ThreadPool.Names.SEARCH).execute(() -> {
                try {
                    cacheFileReader.read(this, ByteBuffer.allocate(blob.length), 0, blob.length, blob.length, "test-search");
                    searchFuture.onResponse(null);
                } catch (Exception e) {
                    searchFuture.onFailure(e);
                }
            });
            safeGet(searchFuture);
            assertSearchOriginMeasurementAttribute(meterRegistry, CachePopulationSource.BlobStore.name());
            meterRegistry.getRecorder().resetCalls();

            // VBCC (peer/index-tier) thread → Peer attribute
            PlainActionFuture<Void> vbccFuture = new PlainActionFuture<>();
            threadPool.executor(StatelessPlugin.GET_VIRTUAL_BATCHED_COMPOUND_COMMIT_CHUNK_THREAD_POOL).execute(() -> {
                try {
                    cacheFileReader.read(this, ByteBuffer.allocate(blob.length), 0, blob.length, blob.length, "test-vbcc");
                    vbccFuture.onResponse(null);
                } catch (Exception e) {
                    vbccFuture.onFailure(e);
                }
            });
            safeGet(vbccFuture);
            assertSearchOriginMeasurementAttribute(meterRegistry, CachePopulationSource.Peer.name());
            meterRegistry.getRecorder().resetCalls();

            // Non-EsThread (plain JUnit thread) → no metric recorded
            cacheFileReader.read(this, ByteBuffer.allocate(blob.length), 0, blob.length, blob.length, "test-plain");
            assertThat(
                meterRegistry.getRecorder()
                    .getMeasurements(InstrumentType.LONG_HISTOGRAM, BlobCacheMetrics.SEARCH_ORIGIN_REMOTE_STORAGE_DOWNLOAD_TOOK_TIME)
                    .size(),
                equalTo(0)
            );
        }
    }

    /**
     * A miss is reported as {@link Outcome#FETCHING} and starts exactly one fetch; once that fetch has landed the same
     * range is {@link Outcome#RESIDENT} and nothing is fetched again. The fetch executor is direct, so the fetch has
     * completed by the time the first call returns; the outcome is still FETCHING because it is a snapshot of the state
     * at decision time.
     */
    public void testEnsureResidentFetchesThenResident() throws Exception {
        Settings settings = nodeSettings();
        RecordingMeterRegistry meterRegistry = new RecordingMeterRegistry();
        BlobCacheMetrics metrics = new BlobCacheMetrics(meterRegistry, NOOP_TIME_PROVIDER);

        try (
            NodeEnvironment env = new NodeEnvironment(settings, TestEnvironment.newEnvironment(settings));
            StatelessSharedBlobCacheService service = newCacheService(env, settings, threadPool)
        ) {
            String fileName = "ensure-resident-fetch";
            byte[] blob = randomByteArrayOfLength(BLOB_LENGTH);
            FileCacheKey cacheKey = new FileCacheKey(new ShardId(new Index("idx", "uid"), 0), 1L, fileName);
            AtomicInteger fetchCount = new AtomicInteger();
            PrefetchBudget budget = new PrefetchBudget(randomIntBetween(1, 8));
            CacheFileReader cacheFileReader = newReader(
                service,
                cacheKey,
                blob,
                countingObjectStoreReader(fileName, blob, service.getRangeSize(), fetchCount),
                metrics,
                true,
                budget
            );

            long offset = randomLongBetween(0, blob.length - 2);
            long length = randomLongBetween(1, blob.length - offset);
            assertThat(cacheFileReader.ensureResident(offset, length), equalTo(Outcome.FETCHING));
            assertThat("a miss must fetch exactly one region", fetchCount.get(), equalTo(1));
            assertPrefetchMetric(meterRegistry, BlobCacheMetrics.PrefetchResult.Fetched, 1);
            assertPrefetchMetric(meterRegistry, BlobCacheMetrics.PrefetchResult.Failed, 0);
            assertThat("the budget slot must be returned once the fetch settles", budget.inFlightRegions(), equalTo(0));

            assertThat(cacheFileReader.ensureResident(offset, length), equalTo(Outcome.RESIDENT));
            assertThat("a resident range must not fetch", fetchCount.get(), equalTo(1));
            assertPrefetchMetric(meterRegistry, BlobCacheMetrics.PrefetchResult.AlreadyCached, 1);
            assertThat(cacheFileReader.residencyRegionSize(), equalTo(REGION_SIZE.getBytes()));
        }
    }

    /**
     * While a region's fetch is in flight, a second request for it joins that fetch: it is reported as
     * {@link Outcome#FETCHING}, does not start another fetch and does not take another budget slot.
     */
    public void testEnsureResidentJoinsInFlightFetch() throws Exception {
        Settings settings = nodeSettings();
        BlobCacheMetrics metrics = new BlobCacheMetrics(new RecordingMeterRegistry(), NOOP_TIME_PROVIDER);

        try (
            NodeEnvironment env = new NodeEnvironment(settings, TestEnvironment.newEnvironment(settings));
            StatelessSharedBlobCacheService service = newCacheService(env, settings, threadPool)
        ) {
            String fileName = "ensure-resident-join";
            byte[] blob = randomByteArrayOfLength(BLOB_LENGTH);
            FileCacheKey cacheKey = new FileCacheKey(new ShardId(new Index("idx", "uid"), 0), 1L, fileName);
            AtomicInteger fetchCount = new AtomicInteger();
            CountDownLatch gate = new CountDownLatch(1);
            PrefetchBudget budget = new PrefetchBudget(randomIntBetween(1, 8));
            CacheFileReader cacheFileReader = newReader(
                service,
                cacheKey,
                blob,
                gatedObjectStoreReader(fileName, blob, service.getRangeSize(), fetchCount, gate),
                metrics,
                true,
                budget
            );

            assertThat(cacheFileReader.ensureResident(0L, blob.length), equalTo(Outcome.FETCHING));
            assertThat(budget.inFlightRegions(), equalTo(1));
            // a copy of the reader (a clone or slice of the index input) shares the budget and the in-flight region
            assertThat(cacheFileReader.copy().ensureResident(0L, blob.length), equalTo(Outcome.FETCHING));
            assertThat("joining must not take another slot", budget.inFlightRegions(), equalTo(1));
            assertBusy(() -> assertThat("exactly one fetch must have been issued", fetchCount.get(), equalTo(1)));

            gate.countDown();
            assertBusy(() -> assertThat(cacheFileReader.ensureResident(0L, blob.length), equalTo(Outcome.RESIDENT)));
            assertBusy(() -> assertThat(budget.inFlightRegions(), equalTo(0)));
            assertThat("joining must not have fetched again", fetchCount.get(), equalTo(1));
        }
    }

    /**
     * Joining an in-flight region must still cover the caller's bytes. The budget tracks regions, but a fetch may cover
     * only part of a region when the blob reader's range is smaller than the region, as when reading from an indexing
     * node in chunks. Here the reader fetches one page at a time: the first request fetches page 0, and a joined request
     * for page 2 of the same region must issue its own fetch rather than trust the region key, while still taking no
     * budget slot.
     */
    public void testEnsureResidentJoinedRegionStillFetchesRequestedBytes() throws Exception {
        Settings settings = nodeSettings();
        BlobCacheMetrics metrics = new BlobCacheMetrics(new RecordingMeterRegistry(), NOOP_TIME_PROVIDER);

        try (
            NodeEnvironment env = new NodeEnvironment(settings, TestEnvironment.newEnvironment(settings));
            StatelessSharedBlobCacheService service = newCacheService(env, settings, threadPool)
        ) {
            String fileName = "ensure-resident-join-partial";
            byte[] blob = randomByteArrayOfLength(BLOB_LENGTH); // one region of REGION_PAGES pages
            FileCacheKey cacheKey = new FileCacheKey(new ShardId(new Index("idx", "uid"), 0), 1L, fileName);
            AtomicInteger fetchCount = new AtomicInteger();
            CountDownLatch gate = new CountDownLatch(1);
            PrefetchBudget budget = new PrefetchBudget(randomIntBetween(1, 8));
            // page-sized ranges, so each fetch covers a single page of the region rather than the whole region
            CacheFileReader cacheFileReader = newReader(
                service,
                cacheKey,
                blob,
                gatedObjectStoreReader(fileName, blob, SharedBytes.PAGE_SIZE, fetchCount, gate),
                metrics,
                true,
                budget
            );

            assertThat(cacheFileReader.ensureResident(0L, 1L), equalTo(Outcome.FETCHING));
            assertThat(budget.inFlightRegions(), equalTo(1));
            assertBusy(() -> assertThat(fetchCount.get(), equalTo(1)));

            long otherPage = 2L * SharedBytes.PAGE_SIZE;
            assertThat(
                "another sub-range of the in-flight region joins",
                cacheFileReader.ensureResident(otherPage, 1L),
                equalTo(Outcome.FETCHING)
            );
            assertThat("joining must not take another slot", budget.inFlightRegions(), equalTo(1));
            assertBusy(() -> assertThat("the joined request must fetch its own bytes", fetchCount.get(), equalTo(2)));

            gate.countDown();
            assertBusy(() -> assertThat(budget.inFlightRegions(), equalTo(0)));
            assertBusy(() -> assertThat(cacheFileReader.ensureResident(otherPage, 1L), equalTo(Outcome.RESIDENT)));
            assertThat(cacheFileReader.ensureResident(0L, 1L), equalTo(Outcome.RESIDENT));
            assertThat("no further fetch once both pages are cached", fetchCount.get(), equalTo(2));
        }
    }

    /**
     * With the budget full, a miss on a different region is {@link Outcome#SKIPPED} and starts no fetch. Once a slot is
     * released the same request is admitted.
     */
    public void testEnsureResidentSkippedWhenBudgetExhausted() throws Exception {
        Settings settings = nodeSettings();
        RecordingMeterRegistry meterRegistry = new RecordingMeterRegistry();
        BlobCacheMetrics metrics = new BlobCacheMetrics(meterRegistry, NOOP_TIME_PROVIDER);

        try (
            NodeEnvironment env = new NodeEnvironment(settings, TestEnvironment.newEnvironment(settings));
            StatelessSharedBlobCacheService service = newCacheService(env, settings, threadPool)
        ) {
            String fileName = "ensure-resident-budget";
            byte[] blob = randomByteArrayOfLength(2 * BLOB_LENGTH); // two regions
            FileCacheKey cacheKey = new FileCacheKey(new ShardId(new Index("idx", "uid"), 0), 1L, fileName);
            AtomicInteger fetchCount = new AtomicInteger();
            CountDownLatch gate = new CountDownLatch(1);
            PrefetchBudget budget = new PrefetchBudget(1);
            CacheFileReader cacheFileReader = newReader(
                service,
                cacheKey,
                blob,
                gatedObjectStoreReader(fileName, blob, service.getRangeSize(), fetchCount, gate),
                metrics,
                true,
                budget
            );

            assertThat(cacheFileReader.ensureResident(0L, 1L), equalTo(Outcome.FETCHING));
            assertThat(budget.inFlightRegions(), equalTo(1));
            assertThat(cacheFileReader.ensureResident(BLOB_LENGTH, 1L), equalTo(Outcome.SKIPPED));
            assertBusy(() -> assertThat("a skipped region must not be fetched", fetchCount.get(), equalTo(1)));
            assertPrefetchMetric(meterRegistry, BlobCacheMetrics.PrefetchResult.Skipped, 1);

            gate.countDown();
            assertBusy(() -> assertThat(budget.inFlightRegions(), equalTo(0)));
            assertThat(cacheFileReader.ensureResident(BLOB_LENGTH, 1L), equalTo(Outcome.FETCHING));
            assertBusy(() -> assertThat(fetchCount.get(), equalTo(2)));
            assertBusy(() -> assertThat(cacheFileReader.ensureResident(BLOB_LENGTH, 1L), equalTo(Outcome.RESIDENT)));
        }
    }

    /**
     * With object store prefetch disabled a miss is {@link Outcome#SKIPPED} and nothing is fetched, but data that a
     * regular read brought into the cache is still reported {@link Outcome#RESIDENT}.
     */
    public void testEnsureResidentSkippedWhenObjectStorePrefetchDisabled() throws Exception {
        Settings settings = nodeSettings();
        RecordingMeterRegistry meterRegistry = new RecordingMeterRegistry();
        BlobCacheMetrics metrics = new BlobCacheMetrics(meterRegistry, NOOP_TIME_PROVIDER);

        try (
            NodeEnvironment env = new NodeEnvironment(settings, TestEnvironment.newEnvironment(settings));
            StatelessSharedBlobCacheService service = newCacheService(env, settings, threadPool)
        ) {
            String fileName = "ensure-resident-disabled";
            byte[] blob = randomByteArrayOfLength(BLOB_LENGTH);
            FileCacheKey cacheKey = new FileCacheKey(new ShardId(new Index("idx", "uid"), 0), 1L, fileName);
            AtomicInteger fetchCount = new AtomicInteger();
            CacheFileReader cacheFileReader = newReader(
                service,
                cacheKey,
                blob,
                countingObjectStoreReader(fileName, blob, service.getRangeSize(), fetchCount),
                metrics,
                false,
                PrefetchBudget.UNLIMITED
            );

            assertThat(cacheFileReader.ensureResident(0L, blob.length), equalTo(Outcome.SKIPPED));
            assertThat(fetchCount.get(), equalTo(0));
            assertPrefetchMetric(meterRegistry, BlobCacheMetrics.PrefetchResult.Skipped, 1);

            cacheFileReader.read(this, ByteBuffer.allocate(blob.length), 0, blob.length, blob.length, "test-plain");
            assertThat(cacheFileReader.ensureResident(0L, blob.length), equalTo(Outcome.RESIDENT));
            assertPrefetchMetric(meterRegistry, BlobCacheMetrics.PrefetchResult.Skipped, 1);
        }
    }

    /**
     * Requests outside the blob cannot be made resident and are {@link Outcome#SKIPPED} without touching the object
     * store. The index input rejects such ranges earlier; this guards the reader on its own.
     */
    public void testEnsureResidentOutsideBlobIsSkipped() throws Exception {
        Settings settings = nodeSettings();
        BlobCacheMetrics metrics = new BlobCacheMetrics(new RecordingMeterRegistry(), NOOP_TIME_PROVIDER);

        try (
            NodeEnvironment env = new NodeEnvironment(settings, TestEnvironment.newEnvironment(settings));
            StatelessSharedBlobCacheService service = newCacheService(env, settings, threadPool)
        ) {
            String fileName = "ensure-resident-eof";
            byte[] blob = randomByteArrayOfLength(BLOB_LENGTH);
            FileCacheKey cacheKey = new FileCacheKey(new ShardId(new Index("idx", "uid"), 0), 1L, fileName);
            AtomicInteger fetchCount = new AtomicInteger();
            CacheFileReader cacheFileReader = newReader(
                service,
                cacheKey,
                blob,
                countingObjectStoreReader(fileName, blob, service.getRangeSize(), fetchCount),
                metrics,
                true,
                PrefetchBudget.UNLIMITED
            );

            long pastEof = randomBoolean() ? blob.length : blob.length + randomLongBetween(1L, 1024L);
            assertThat(cacheFileReader.ensureResident(pastEof, randomLongBetween(1L, 1024L)), equalTo(Outcome.SKIPPED));
            assertThat(cacheFileReader.ensureResident(0L, randomBoolean() ? 0L : -randomLongBetween(1L, 1024L)), equalTo(Outcome.SKIPPED));
            assertThat(fetchCount.get(), equalTo(0));
        }
    }

    /**
     * The bulk form evaluates each touched region once and hands every range in that region the region's outcome:
     * ranges in the resident region are {@link Outcome#RESIDENT}, ranges in the missing region are all
     * {@link Outcome#FETCHING}, and that missing region is fetched exactly once however many ranges land in it.
     */
    public void testEnsureResidentBulkGroupsByRegion() throws Exception {
        Settings settings = nodeSettings();
        BlobCacheMetrics metrics = new BlobCacheMetrics(new RecordingMeterRegistry(), NOOP_TIME_PROVIDER);

        try (
            NodeEnvironment env = new NodeEnvironment(settings, TestEnvironment.newEnvironment(settings));
            StatelessSharedBlobCacheService service = newCacheService(env, settings, threadPool)
        ) {
            String fileName = "ensure-resident-bulk";
            byte[] blob = randomByteArrayOfLength(3 * BLOB_LENGTH); // three regions
            FileCacheKey cacheKey = new FileCacheKey(new ShardId(new Index("idx", "uid"), 0), 1L, fileName);
            AtomicInteger fetchCount = new AtomicInteger();
            PrefetchBudget budget = new PrefetchBudget(randomIntBetween(1, 8));
            CacheFileReader cacheFileReader = newReader(
                service,
                cacheKey,
                blob,
                countingObjectStoreReader(fileName, blob, service.getRangeSize(), fetchCount),
                metrics,
                true,
                budget
            );

            // make region 0 resident through a regular read; regions 1 and 2 stay absent
            cacheFileReader.read(this, ByteBuffer.allocate(BLOB_LENGTH), 0, BLOB_LENGTH, blob.length, "test-plain");
            int fetchesAfterRead = fetchCount.get();

            final int recordLength = 64;
            final int perRegion = randomIntBetween(2, 10);
            final int count = 2 * perRegion;
            final long[] offsets = new long[count + randomIntBetween(0, 3)]; // may be longer than count
            for (int i = 0; i < perRegion; i++) {
                offsets[i] = randomLongBetween(0, BLOB_LENGTH - recordLength);
                offsets[perRegion + i] = 2L * BLOB_LENGTH + randomLongBetween(0, BLOB_LENGTH - recordLength);
            }
            final Outcome[] outcomes = new Outcome[offsets.length];

            cacheFileReader.ensureResident(offsets, recordLength, count, outcomes);
            for (int i = 0; i < perRegion; i++) {
                assertThat("range in resident region 0: " + i, outcomes[i], equalTo(Outcome.RESIDENT));
                assertThat("range in missing region 2: " + i, outcomes[perRegion + i], equalTo(Outcome.FETCHING));
            }
            for (int i = count; i < outcomes.length; i++) {
                assertNull("entries beyond count must not be written", outcomes[i]);
            }
            assertThat("region 2 must be fetched exactly once, region 1 not at all", fetchCount.get(), equalTo(fetchesAfterRead + 1));
            assertThat(budget.inFlightRegions(), equalTo(0));

            cacheFileReader.ensureResident(offsets, recordLength, count, outcomes);
            for (int i = 0; i < count; i++) {
                assertThat(outcomes[i], equalTo(Outcome.RESIDENT));
            }
            assertThat(fetchCount.get(), equalTo(fetchesAfterRead + 1));
        }
    }

    /**
     * A range spanning a resident region and a missing one reports the more expensive outcome: SKIPPED when the
     * missing region cannot be admitted, FETCHING when it can.
     */
    public void testEnsureResidentStraddlingRegionsTakesMostExpensiveOutcome() throws Exception {
        Settings settings = nodeSettings();
        BlobCacheMetrics metrics = new BlobCacheMetrics(new RecordingMeterRegistry(), NOOP_TIME_PROVIDER);

        try (
            NodeEnvironment env = new NodeEnvironment(settings, TestEnvironment.newEnvironment(settings));
            StatelessSharedBlobCacheService service = newCacheService(env, settings, threadPool)
        ) {
            String fileName = "ensure-resident-straddle";
            byte[] blob = randomByteArrayOfLength(2 * BLOB_LENGTH); // two regions
            FileCacheKey cacheKey = new FileCacheKey(new ShardId(new Index("idx", "uid"), 0), 1L, fileName);
            AtomicInteger fetchCount = new AtomicInteger();
            CacheBlobReader reader = countingObjectStoreReader(fileName, blob, service.getRangeSize(), fetchCount);
            CacheFileReader denied = newReader(service, cacheKey, blob, reader, metrics, true, new PrefetchBudget(0));
            CacheFileReader admitted = newReader(service, cacheKey, blob, reader, metrics, true, PrefetchBudget.UNLIMITED);

            // region 0 resident, region 1 absent
            denied.read(this, ByteBuffer.allocate(BLOB_LENGTH), 0, BLOB_LENGTH, blob.length, "test-plain");
            int fetchesAfterRead = fetchCount.get();

            long start = randomLongBetween(0, BLOB_LENGTH - 1);
            long end = BLOB_LENGTH + randomLongBetween(1, BLOB_LENGTH);
            assertThat(denied.ensureResident(start, end - start), equalTo(Outcome.SKIPPED));
            assertThat(fetchCount.get(), equalTo(fetchesAfterRead));

            assertThat(admitted.ensureResident(start, end - start), equalTo(Outcome.FETCHING));
            assertThat(fetchCount.get(), equalTo(fetchesAfterRead + 1));
            assertThat(admitted.ensureResident(start, end - start), equalTo(Outcome.RESIDENT));

            // the bulk form applies the same rule to a straddling record
            Outcome[] outcomes = new Outcome[1];
            admitted.ensureResident(new long[] { start }, Math.toIntExact(end - start), 1, outcomes);
            assertThat(outcomes[0], equalTo(Outcome.RESIDENT));
        }
    }

    private static CacheFileReader newReader(
        StatelessSharedBlobCacheService service,
        FileCacheKey cacheKey,
        byte[] blob,
        CacheBlobReader reader,
        BlobCacheMetrics metrics,
        boolean objectStorePrefetchEnabled,
        PrefetchBudget budget
    ) {
        return new CacheFileReader(
            service.getCacheFile(cacheKey, blob.length, SharedBlobCacheService.CacheMissHandler.NOOP, randomRegionTimestampMillis()),
            reader,
            createBlobFileRanges(1L, 0L, 0, blob.length),
            metrics,
            System::currentTimeMillis,
            service.getRegionSize(),
            IOContext.DEFAULT,
            false,
            objectStorePrefetchEnabled,
            budget
        );
    }

    /**
     * Like {@link #countingObjectStoreReader} but runs the fetch on the shard-read pool and holds it until {@code gate}
     * opens, so a test can observe the in-flight state. The direct executor used elsewhere completes fetches before the
     * call returns, which makes "in flight" unobservable.
     */
    private CacheBlobReader gatedObjectStoreReader(
        String fileName,
        byte[] blob,
        long cacheRangeSize,
        AtomicInteger counter,
        CountDownLatch gate
    ) {
        return new ObjectStoreCacheBlobReader(
            TestUtils.singleBlobContainer(fileName, blob),
            fileName,
            cacheRangeSize,
            threadPool.executor(StatelessPlugin.SHARD_READ_THREAD_POOL)
        ) {
            @Override
            protected InputStream getRangeInputStream(long position, int length) throws java.io.IOException {
                counter.incrementAndGet();
                safeAwait(gate);
                return super.getRangeInputStream(position, length);
            }
        };
    }

    private static CacheBlobReader countingObjectStoreReader(String fileName, byte[] blob, long cacheRangeSize, AtomicInteger counter) {
        return new ObjectStoreCacheBlobReader(
            TestUtils.singleBlobContainer(fileName, blob),
            fileName,
            cacheRangeSize,
            EsExecutors.DIRECT_EXECUTOR_SERVICE
        ) {
            @Override
            public void getRangeInputStream(long position, int length, ActionListener<InputStream> listener) {
                counter.incrementAndGet();
                super.getRangeInputStream(position, length, listener);
            }
        };
    }

    private static CacheBlobReader alreadyUploadedThenServingReader(
        String fileName,
        byte[] blob,
        long cacheRangeSize,
        int failCount,
        AtomicInteger fetchCount
    ) {
        return new ObjectStoreCacheBlobReader(
            TestUtils.singleBlobContainer(fileName, blob),
            fileName,
            cacheRangeSize,
            EsExecutors.DIRECT_EXECUTOR_SERVICE
        ) {
            @Override
            public void getRangeInputStream(long position, int length, ActionListener<InputStream> listener) {
                if (fetchCount.getAndIncrement() < failCount) {
                    listener.onFailure(new ResourceAlreadyUploadedException("VBCC already uploaded: " + position + "+" + length));
                } else {
                    super.getRangeInputStream(position, length, listener);
                }
            }
        };
    }

    private Settings nodeSettings() {
        return Settings.builder()
            .put(Environment.PATH_HOME_SETTING.getKey(), createTempDir().toAbsolutePath())
            .putList(Environment.PATH_DATA_SETTING.getKey(), createTempDir().toAbsolutePath().toString())
            .put(
                SharedBlobCacheService.SHARED_CACHE_SIZE_SETTING.getKey(),
                ByteSizeValue.ofBytes(50 * SharedBytes.PAGE_SIZE).getStringRep()
            )
            .put(SharedBlobCacheService.SHARED_CACHE_REGION_SIZE_SETTING.getKey(), REGION_SIZE.getStringRep())
            .put(SharedBlobCacheService.SHARED_CACHE_RANGE_SIZE_SETTING.getKey(), REGION_SIZE.getStringRep())
            .put(SharedBlobCacheService.SHARED_CACHE_MMAP.getKey(), true)
            .build();
    }

    private static void assertSearchOriginMeasurementAttribute(RecordingMeterRegistry meterRegistry, String expectedSource) {
        List<Measurement> measurements = meterRegistry.getRecorder()
            .getMeasurements(InstrumentType.LONG_HISTOGRAM, BlobCacheMetrics.SEARCH_ORIGIN_REMOTE_STORAGE_DOWNLOAD_TOOK_TIME);
        assertThat("expected at least one search-origin measurement", measurements.size(), greaterThan(0));
        for (Measurement measurement : measurements) {
            assertThat(measurement.attributes().get(BlobCacheMetrics.CACHE_POPULATION_SOURCE_ATTRIBUTE_KEY), equalTo(expectedSource));
        }
    }

    private static void assertPrefetchMetric(RecordingMeterRegistry meterRegistry, BlobCacheMetrics.PrefetchResult result, long expected) {
        long observed = meterRegistry.getRecorder()
            .getMeasurements(InstrumentType.LONG_COUNTER, BlobCacheMetrics.BLOB_CACHE_PREFETCH_TOTAL)
            .stream()
            .filter(m -> result.name().equals(m.attributes().get(BlobCacheMetrics.PREFETCH_RESULT_ATTRIBUTE_KEY)))
            .mapToLong(Measurement::getLong)
            .sum();
        assertEquals("expected " + expected + " [" + result + "] prefetch measurement(s)", expected, observed);
    }
}
