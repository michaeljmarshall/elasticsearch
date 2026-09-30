/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the "Elastic License
 * 2.0", the "GNU Affero General Public License v3.0 only", and the "Server Side
 * Public License v 1"; you may not use this file except in compliance with, at
 * your election, the "Elastic License 2.0", the "GNU Affero General Public
 * License v3.0 only", or the "Server Side Public License, v 1".
 */

package org.elasticsearch.search.vectors;

import org.apache.lucene.codecs.KnnVectorsFormat;
import org.apache.lucene.codecs.lucene99.Lucene99HnswVectorsFormat;
import org.apache.lucene.document.Document;
import org.apache.lucene.document.KnnFloatVectorField;
import org.apache.lucene.index.DirectoryReader;
import org.apache.lucene.index.FilterDirectoryReader;
import org.apache.lucene.index.FilterLeafReader;
import org.apache.lucene.index.FloatVectorValues;
import org.apache.lucene.index.IndexReader;
import org.apache.lucene.index.IndexWriter;
import org.apache.lucene.index.IndexWriterConfig;
import org.apache.lucene.index.LeafReader;
import org.apache.lucene.index.LeafReaderContext;
import org.apache.lucene.index.NoMergePolicy;
import org.apache.lucene.index.ReaderUtil;
import org.apache.lucene.index.VectorSimilarityFunction;
import org.apache.lucene.queries.function.FunctionScoreQuery;
import org.apache.lucene.search.ConjunctionUtils;
import org.apache.lucene.search.DocIdSetIterator;
import org.apache.lucene.search.DoubleValuesSource;
import org.apache.lucene.search.FieldExistsQuery;
import org.apache.lucene.search.FullPrecisionFloatVectorSimilarityValuesSource;
import org.apache.lucene.search.IndexSearcher;
import org.apache.lucene.search.KnnFloatVectorQuery;
import org.apache.lucene.search.Query;
import org.apache.lucene.search.QueryVisitor;
import org.apache.lucene.search.ScoreDoc;
import org.apache.lucene.search.ScoreMode;
import org.apache.lucene.search.TopDocs;
import org.apache.lucene.search.VectorScorer;
import org.apache.lucene.search.Weight;
import org.apache.lucene.store.Directory;
import org.apache.lucene.store.FilterDirectory;
import org.apache.lucene.store.IOContext;
import org.apache.lucene.store.IndexInput;
import org.apache.lucene.util.Bits;
import org.elasticsearch.common.lucene.search.Queries;
import org.elasticsearch.core.TieredPrefetchInput;
import org.elasticsearch.index.codec.bwc.Elasticsearch93Lucene104Codec;
import org.elasticsearch.index.codec.vectors.es93.ES93HnswScalarQuantizedVectorsFormat;
import org.elasticsearch.index.codec.zstd.Zstd814StoredFieldsFormat;
import org.elasticsearch.index.mapper.vectors.DenseVectorFieldMapper;
import org.elasticsearch.search.profile.query.QueryProfiler;
import org.elasticsearch.search.profile.query.TieredPrefetchOutcomeCounts;
import org.elasticsearch.test.ESTestCase;

import java.io.IOException;
import java.io.UnsupportedEncodingException;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.List;
import java.util.Map;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.function.Function;
import java.util.stream.Collectors;

import static org.elasticsearch.index.codec.vectors.VectorTestUtils.randomFloatVector;
import static org.elasticsearch.index.codec.vectors.diskbbq.ES920DiskBBQVectorsFormat.DEFAULT_CENTROIDS_PER_PARENT_CLUSTER;
import static org.elasticsearch.index.codec.vectors.diskbbq.ES920DiskBBQVectorsFormat.DEFAULT_VECTORS_PER_CLUSTER;
import static org.hamcrest.Matchers.arrayWithSize;
import static org.hamcrest.Matchers.closeTo;
import static org.hamcrest.Matchers.empty;
import static org.hamcrest.Matchers.equalTo;
import static org.hamcrest.Matchers.greaterThan;
import static org.hamcrest.Matchers.hasItem;
import static org.hamcrest.Matchers.hasSize;
import static org.hamcrest.Matchers.lessThanOrEqualTo;
import static org.hamcrest.Matchers.not;

public class RescoreKnnVectorQueryTests extends ESTestCase {

    public static final String FIELD_NAME = "float_vector";

    /*
     * Original KNN scoring and rescoring can use slightly different calculation methods,
     * so there may be a very slight difference in the scores after rescoring.
     */
    private static final float DELTA = 1e-6f;

    public void testRescoreDocs() throws Exception {
        int numDocs = randomIntBetween(10, 200);
        int numDims = randomIntBetween(5, 100);
        int k = randomIntBetween(1, numDocs - 1);

        float[] queryVector = randomFloatVector(numDims);
        List<Query> innerQueries = new ArrayList<>();
        innerQueries.add(
            new KnnFloatVectorQuery(FIELD_NAME, randomFloatVector(numDims), (int) (k * randomFloatBetween(1.0f, 10.0f, true)))
        );
        innerQueries.add(DenseVectorQuery.Floats.codecScored(queryVector, FIELD_NAME).filteredBy(new FieldExistsQuery(FIELD_NAME)));
        innerQueries.add(Queries.ALL_DOCS_INSTANCE);

        try (Directory d = newDirectory()) {
            addRandomDocuments(numDocs, d, numDims);

            try (IndexReader reader = DirectoryReader.open(d)) {
                for (Query innerQuery : innerQueries) {
                    RescoreKnnVectorQuery rescoreKnnVectorQuery = RescoreKnnVectorQuery.fromInnerQuery(
                        FIELD_NAME,
                        queryVector,
                        k,
                        k,
                        innerQuery
                    );

                    IndexSearcher searcher = newSearcher(reader, true, false);

                    TopDocs rescoredDocs = searcher.search(rescoreKnnVectorQuery, numDocs);
                    assertThat(rescoredDocs.scoreDocs, arrayWithSize(k));

                    if (innerQuery instanceof KnnFloatVectorQuery) {
                        // check that at least one doc has its score changed, indicating rescoring has happened
                        TopDocs unrescoredDocs = searcher.search(innerQuery, numDocs);
                        Map<Integer, Double> rescoredScores = Arrays.stream(rescoredDocs.scoreDocs)
                            .collect(Collectors.toMap(sd -> sd.doc, sd -> (double) sd.score));

                        boolean changed = false;
                        for (ScoreDoc unrescored : unrescoredDocs.scoreDocs) {
                            Double rescored = rescoredScores.get(unrescored.doc);
                            if (rescored != null && Math.abs(rescored - unrescored.score) > DELTA) {
                                changed = true;
                                break;
                            }
                        }
                        assertTrue("No docs had their scores changed", changed);
                    }

                    assertScoresMatchGroundTruth(queryVector, searcher, rescoredDocs, numDocs);
                }
            }
        }
    }

    public void testRescoreWithNoMatches() throws Exception {
        int numDocs = randomIntBetween(10, 50);
        int numDims = randomIntBetween(5, 20);
        int k = randomIntBetween(1, numDocs - 1);
        float[] queryVector = randomFloatVector(numDims);

        try (Directory d = newDirectory()) {
            addRandomDocuments(numDocs, d, numDims);

            try (IndexReader reader = DirectoryReader.open(d)) {
                IndexSearcher searcher = newSearcher(reader, true, false);

                // MatchNoDocsQuery triggers the early exit in DirectRescoreKnnVectorQuery.rewrite
                Query noDocsRescore = RescoreKnnVectorQuery.fromInnerQuery(FIELD_NAME, queryVector, k, k, Queries.NO_DOCS_INSTANCE);
                assertThat(searcher.search(noDocsRescore, numDocs).scoreDocs, arrayWithSize(0));

                // A filter that excludes all docs results in an empty conjunction
                Query nonExistentField = new FieldExistsQuery("no_such_field");
                Query emptyRescore = RescoreKnnVectorQuery.fromInnerQuery(FIELD_NAME, queryVector, k, k, nonExistentField);
                assertThat(searcher.search(emptyRescore, numDocs).scoreDocs, arrayWithSize(0));
            }
        }
    }

    // Tests rescoring with doc counts exceeding several bulk scoring batches (32) per leaf.
    // Also exercises {@code rescoreK > k} which routes through the {@code LateRescoreQuery} path.
    public void testRescoreWithLargeDocCount() throws Exception {
        int numDocs = randomIntBetween(200, 500);
        int numDims = randomIntBetween(5, 50);
        int k = randomIntBetween(1, 10);
        int rescoreK = randomIntBetween(k + 1, numDocs);

        float[] queryVector = randomFloatVector(numDims);

        try (Directory d = newDirectory()) {
            addRandomDocuments(numDocs, d, numDims);

            try (IndexReader reader = DirectoryReader.open(d)) {
                RescoreKnnVectorQuery rescoreKnnVectorQuery = RescoreKnnVectorQuery.fromInnerQuery(
                    FIELD_NAME,
                    queryVector,
                    k,
                    rescoreK,
                    Queries.ALL_DOCS_INSTANCE
                );

                IndexSearcher searcher = newSearcher(reader, true, false);
                TopDocs rescoredDocs = searcher.search(rescoreKnnVectorQuery, numDocs);
                assertThat(rescoredDocs.scoreDocs, arrayWithSize(k));

                assertScoresMatchGroundTruth(queryVector, searcher, rescoredDocs, numDocs);
            }
        }
    }

    // Verifies that rescored results appear in the same order and with the same scores
    // as a full-precision brute-force cosine similarity search over all documents.
    private static void assertScoresMatchGroundTruth(float[] queryVector, IndexSearcher searcher, TopDocs rescoredDocs, int numDocs)
        throws IOException {
        DoubleValuesSource valueSource = new FullPrecisionFloatVectorSimilarityValuesSource(
            queryVector,
            FIELD_NAME,
            VectorSimilarityFunction.COSINE
        );
        FunctionScoreQuery functionScoreQuery = new FunctionScoreQuery(Queries.ALL_DOCS_INSTANCE, valueSource);
        TopDocs realScoreTopDocs = searcher.search(functionScoreQuery, numDocs);

        int i = 0;
        ScoreDoc[] realScoreDocs = realScoreTopDocs.scoreDocs;
        for (ScoreDoc rescoreDoc : rescoredDocs.scoreDocs) {
            // There are docs that won't be found in the rescored search, but every doc found must be in the same order
            // and have the same score
            while (i < realScoreDocs.length && realScoreDocs[i].doc != rescoreDoc.doc) {
                i++;
            }
            if (i >= realScoreDocs.length) {
                fail("Rescored doc not found in real score docs");
            }
            assertThat("Real score is not the same as rescored score", (double) rescoreDoc.score, closeTo(realScoreDocs[i].score, DELTA));
        }
    }

    public void testRescoreSingleAndBulkEquality() throws Exception {
        int numDocs = randomIntBetween(10, 100);
        int numDims = randomIntBetween(5, 100);
        int k = randomIntBetween(1, numDocs - 1);

        float[] queryVector = randomFloatVector(numDims);

        List<Query> innerQueries = new ArrayList<>();
        innerQueries.add(
            new KnnFloatVectorQuery(FIELD_NAME, randomFloatVector(numDims), (int) (k * randomFloatBetween(1.0f, 10.0f, true)))
        );
        innerQueries.add(DenseVectorQuery.Floats.codecScored(queryVector, FIELD_NAME).filteredBy(new FieldExistsQuery(FIELD_NAME)));
        innerQueries.add(Queries.ALL_DOCS_INSTANCE);

        try (Directory d = newDirectory()) {
            addRandomDocuments(numDocs, d, numDims);
            try (DirectoryReader reader = DirectoryReader.open(d)) {
                for (Query innerQuery : innerQueries) {
                    RescoreKnnVectorQuery rescoreKnnVectorQuery = RescoreKnnVectorQuery.fromInnerQuery(
                        FIELD_NAME,
                        queryVector,
                        k,
                        k,
                        innerQuery
                    );

                    IndexSearcher searcher = newSearcher(reader, true, false);
                    TopDocs rescoredDocs = searcher.search(rescoreKnnVectorQuery, numDocs);
                    assertThat(rescoredDocs.scoreDocs, arrayWithSize(k));

                    searcher = newSearcher(new SingleVectorQueryIndexReader(reader), true, false);
                    rescoreKnnVectorQuery = RescoreKnnVectorQuery.fromInnerQuery(FIELD_NAME, queryVector, k, k, innerQuery);
                    TopDocs singleRescored = searcher.search(rescoreKnnVectorQuery, numDocs);
                    assertThat(singleRescored.scoreDocs, arrayWithSize(k));

                    // Get real scores
                    ScoreDoc[] singleRescoreDocs = singleRescored.scoreDocs;
                    int i = 0;
                    for (ScoreDoc rescoreDoc : rescoredDocs.scoreDocs) {
                        assertThat(rescoreDoc.doc, equalTo(singleRescoreDocs[i].doc));
                        assertThat((double) rescoreDoc.score, closeTo(singleRescoreDocs[i].score, DELTA));
                        i++;
                    }
                }
            }
        }
    }

    public void testProfiling() throws Exception {
        int numDocs = randomIntBetween(10, 100);
        int numDims = randomIntBetween(5, 100);
        int k = randomIntBetween(1, numDocs - 1);

        try (Directory d = newDirectory()) {
            addRandomDocuments(numDocs, d, numDims);

            try (IndexReader reader = DirectoryReader.open(d)) {
                float[] queryVector = randomFloatVector(numDims);

                checkProfiling(k, numDocs, queryVector, reader, Queries.ALL_DOCS_INSTANCE);
                checkProfiling(k, numDocs, queryVector, reader, new MockQueryProfilerProvider(randomIntBetween(1, 100)));
            }
        }
    }

    private void checkProfiling(int k, int numDocs, float[] queryVector, IndexReader reader, Query innerQuery) throws IOException {
        var rescoreKnnVectorQuery = RescoreKnnVectorQuery.fromInnerQuery(FIELD_NAME, queryVector, k, k, innerQuery);
        IndexSearcher searcher = newSearcher(reader, true, false);
        searcher.search(rescoreKnnVectorQuery, numDocs);

        QueryProfiler queryProfiler = new QueryProfiler();
        rescoreKnnVectorQuery.profile(queryProfiler);

        long expectedVectorOpsCount = numDocs;
        if (innerQuery instanceof QueryProfilerProvider queryProfilerProvider) {
            QueryProfiler anotherProfiler = new QueryProfiler();
            queryProfilerProvider.profile(anotherProfiler);
            assertThat(anotherProfiler.getVectorOpsCount(), greaterThan(0L));
            expectedVectorOpsCount += anotherProfiler.getVectorOpsCount();
        }

        assertThat(queryProfiler.getVectorOpsCount(), equalTo(expectedVectorOpsCount));
    }

    /**
     * With vector data behind a {@link TieredPrefetchInput}, rescoring makes exactly one bulk
     * {@link TieredPrefetchInput#ensureResident(long[], int, int, TieredPrefetchInput.Outcome[])} call per leaf that has
     * candidates, covering every candidate's vector, counts the mixed outcomes, and returns the same top docs as the
     * plain {@link IndexInput#prefetch} path.
     */
    public void testTieredPrefetchBulkCallPerLeafWithMixedOutcomes() throws Exception {
        int numDocs = randomIntBetween(50, 300);
        int numDims = randomIntBetween(5, 50);
        int vectorByteSize = numDims * Float.BYTES;
        int k = randomIntBetween(1, 10);
        int rescoreK = randomIntBetween(k, numDocs);
        float[] queryVector = randomFloatVector(numDims);
        TieredPrefetchInput.Outcome[] outcomeValues = TieredPrefetchInput.Outcome.values();
        Function<Long, TieredPrefetchInput.Outcome> script = offset -> outcomeValues[(int) ((offset / vectorByteSize) % 3)];

        try (Directory base = newDirectory()) {
            addFlatVectorDocuments(base, numDocs, numDims);
            PrefetchRecorder tiered = new PrefetchRecorder(script);
            PrefetchRecorder plain = new PrefetchRecorder(null);
            try (
                DirectoryReader tieredReader = DirectoryReader.open(new RecordingDirectory(base, tiered));
                DirectoryReader plainReader = DirectoryReader.open(new RecordingDirectory(base, plain))
            ) {
                IndexSearcher tieredSearcher = newSearcher(tieredReader, false, false);
                IndexSearcher plainSearcher = newSearcher(plainReader, false, false);
                List<long[]> expectedOffsets = expectedCandidateOffsets(tieredSearcher, rescoreK, vectorByteSize);
                assertThat(expectedOffsets.isEmpty(), equalTo(false));

                tiered.reset();
                RescoreKnnVectorQuery tieredQuery = RescoreKnnVectorQuery.fromInnerQuery(
                    FIELD_NAME,
                    queryVector,
                    k,
                    rescoreK,
                    Queries.ALL_DOCS_INSTANCE
                );
                TopDocs tieredTopDocs = tieredSearcher.search(tieredQuery, k);

                // exactly one bulk call per leaf with candidates, covering all of them, in leaf order
                assertThat(tiered.bulkCalls, hasSize(expectedOffsets.size()));
                for (int i = 0; i < expectedOffsets.size(); i++) {
                    assertArrayEquals(expectedOffsets.get(i), tiered.bulkCalls.get(i).offsets());
                    assertThat(tiered.bulkCalls.get(i).length(), equalTo(vectorByteSize));
                }
                assertThat(tiered.singleEnsureResidentCalls.get(), equalTo(0));
                // the plain hint is not issued on top of the bulk call
                for (long[] leafOffsets : expectedOffsets) {
                    for (long offset : leafOffsets) {
                        assertThat(tiered.prefetches, not(hasItem(new Range(offset, vectorByteSize))));
                    }
                }

                // outcomes are counted per candidate
                TieredPrefetchOutcomeCounts expectedCounts = new TieredPrefetchOutcomeCounts();
                for (long[] leafOffsets : expectedOffsets) {
                    for (long offset : leafOffsets) {
                        expectedCounts.add(script.apply(offset));
                    }
                }
                assertThat(expectedCounts.total(), equalTo((long) rescoreK));
                assertOutcomeCounts(tieredQuery.tieredPrefetchOutcomes(), expectedCounts);
                QueryProfiler queryProfiler = new QueryProfiler();
                tieredQuery.profile(queryProfiler);
                assertOutcomeCounts(queryProfiler.getTieredPrefetchOutcomes(), expectedCounts);

                // results are identical to the non-tiered path, and correct
                TopDocs plainTopDocs = plainSearcher.search(
                    RescoreKnnVectorQuery.fromInnerQuery(FIELD_NAME, queryVector, k, rescoreK, Queries.ALL_DOCS_INSTANCE),
                    k
                );
                assertSameTopDocs(plainTopDocs, tieredTopDocs);
                assertScoresMatchGroundTruth(queryVector, tieredSearcher, tieredTopDocs, numDocs);
            }
        }
    }

    /**
     * When the tiered input schedules nothing at all, rescoring still reads every candidate and returns correct results.
     */
    public void testTieredPrefetchAllSkipped() throws Exception {
        int numDocs = randomIntBetween(50, 300);
        int numDims = randomIntBetween(5, 50);
        int vectorByteSize = numDims * Float.BYTES;
        int k = randomIntBetween(1, 10);
        int rescoreK = randomIntBetween(k, numDocs);
        float[] queryVector = randomFloatVector(numDims);

        try (Directory base = newDirectory()) {
            addFlatVectorDocuments(base, numDocs, numDims);
            PrefetchRecorder tiered = new PrefetchRecorder(offset -> TieredPrefetchInput.Outcome.SKIPPED);
            PrefetchRecorder plain = new PrefetchRecorder(null);
            try (
                DirectoryReader tieredReader = DirectoryReader.open(new RecordingDirectory(base, tiered));
                DirectoryReader plainReader = DirectoryReader.open(new RecordingDirectory(base, plain))
            ) {
                IndexSearcher tieredSearcher = newSearcher(tieredReader, false, false);
                IndexSearcher plainSearcher = newSearcher(plainReader, false, false);
                List<long[]> expectedOffsets = expectedCandidateOffsets(tieredSearcher, rescoreK, vectorByteSize);

                tiered.reset();
                RescoreKnnVectorQuery tieredQuery = RescoreKnnVectorQuery.fromInnerQuery(
                    FIELD_NAME,
                    queryVector,
                    k,
                    rescoreK,
                    Queries.ALL_DOCS_INSTANCE
                );
                TopDocs tieredTopDocs = tieredSearcher.search(tieredQuery, k);
                assertThat(tiered.bulkCalls, hasSize(expectedOffsets.size()));
                assertThat(tieredTopDocs.scoreDocs, arrayWithSize(k));

                TieredPrefetchOutcomeCounts counts = tieredQuery.tieredPrefetchOutcomes();
                assertThat(counts.skipped(), equalTo((long) rescoreK));
                assertThat(counts.resident(), equalTo(0L));
                assertThat(counts.fetching(), equalTo(0L));

                TopDocs plainTopDocs = plainSearcher.search(
                    RescoreKnnVectorQuery.fromInnerQuery(FIELD_NAME, queryVector, k, rescoreK, Queries.ALL_DOCS_INSTANCE),
                    k
                );
                assertSameTopDocs(plainTopDocs, tieredTopDocs);
                assertScoresMatchGroundTruth(queryVector, tieredSearcher, tieredTopDocs, numDocs);
            }
        }
    }

    /**
     * When the vector data is not behind a {@link TieredPrefetchInput}, every candidate still gets the plain
     * {@link IndexInput#prefetch} hint, no outcomes are counted, and results are correct.
     */
    public void testPrefetchFallbackWithoutTieredInput() throws Exception {
        int numDocs = randomIntBetween(50, 300);
        int numDims = randomIntBetween(5, 50);
        int vectorByteSize = numDims * Float.BYTES;
        int k = randomIntBetween(1, 10);
        int rescoreK = randomIntBetween(k, numDocs);
        float[] queryVector = randomFloatVector(numDims);

        try (Directory base = newDirectory()) {
            addFlatVectorDocuments(base, numDocs, numDims);
            PrefetchRecorder plain = new PrefetchRecorder(null);
            try (DirectoryReader plainReader = DirectoryReader.open(new RecordingDirectory(base, plain))) {
                IndexSearcher plainSearcher = newSearcher(plainReader, false, false);
                List<long[]> expectedOffsets = expectedCandidateOffsets(plainSearcher, rescoreK, vectorByteSize);

                plain.reset();
                RescoreKnnVectorQuery query = RescoreKnnVectorQuery.fromInnerQuery(
                    FIELD_NAME,
                    queryVector,
                    k,
                    rescoreK,
                    Queries.ALL_DOCS_INSTANCE
                );
                TopDocs topDocs = plainSearcher.search(query, k);

                assertThat(plain.bulkCalls, empty());
                for (long[] leafOffsets : expectedOffsets) {
                    for (long offset : leafOffsets) {
                        assertThat(plain.prefetches, hasItem(new Range(offset, vectorByteSize)));
                    }
                }
                assertThat(query.tieredPrefetchOutcomes().total(), equalTo(0L));
                assertThat(topDocs.scoreDocs, arrayWithSize(k));
                assertScoresMatchGroundTruth(queryVector, plainSearcher, topDocs, numDocs);
            }
        }
    }

    private static void assertOutcomeCounts(TieredPrefetchOutcomeCounts actual, TieredPrefetchOutcomeCounts expected) {
        assertThat(actual.resident(), equalTo(expected.resident()));
        assertThat(actual.fetching(), equalTo(expected.fetching()));
        assertThat(actual.skipped(), equalTo(expected.skipped()));
    }

    private static void assertSameTopDocs(TopDocs expected, TopDocs actual) {
        assertThat(actual.scoreDocs, arrayWithSize(expected.scoreDocs.length));
        for (int i = 0; i < expected.scoreDocs.length; i++) {
            assertThat(actual.scoreDocs[i].doc, equalTo(expected.scoreDocs[i].doc));
            assertThat(actual.scoreDocs[i].score, equalTo(expected.scoreDocs[i].score));
        }
    }

    /**
     * The late rescoring path with a match-all inner query rescores the {@code rescoreK} lowest doc IDs (constant scores
     * tie-break on doc ID). Every doc has a vector and nothing is deleted, so a doc's ord is its leaf-relative doc ID.
     * Returns the expected vector offsets per leaf, in leaf order, for leaves that have candidates.
     */
    private static List<long[]> expectedCandidateOffsets(IndexSearcher searcher, int rescoreK, int vectorByteSize) throws IOException {
        TopDocs innerTopDocs = searcher.search(Queries.ALL_DOCS_INSTANCE, rescoreK);
        List<LeafReaderContext> leaves = searcher.getIndexReader().leaves();
        List<List<Long>> perLeaf = new ArrayList<>();
        for (int i = 0; i < leaves.size(); i++) {
            perLeaf.add(new ArrayList<>());
        }
        for (ScoreDoc scoreDoc : innerTopDocs.scoreDocs) {
            int leafIndex = ReaderUtil.subIndex(scoreDoc.doc, leaves);
            LeafReaderContext leaf = leaves.get(leafIndex);
            assertThat(leaf.reader().hasDeletions(), equalTo(false));
            perLeaf.get(leafIndex).add((long) (scoreDoc.doc - leaf.docBase) * vectorByteSize);
        }
        List<long[]> expected = new ArrayList<>();
        for (List<Long> leafOffsets : perLeaf) {
            if (leafOffsets.isEmpty() == false) {
                expected.add(leafOffsets.stream().sorted().mapToLong(Long::longValue).toArray());
            }
        }
        return expected;
    }

    /**
     * Indexes one float vector per doc with the Lucene flat HNSW format, whose vector values expose their data through
     * {@link org.apache.lucene.codecs.lucene95.HasIndexSlice}, which the tiered prefetch path needs. Random commits give
     * several leaves.
     */
    private static void addFlatVectorDocuments(Directory d, int numDocs, int numDims) throws IOException {
        IndexWriterConfig iwc = newIndexWriterConfig();
        iwc.setCodec(new Elasticsearch93Lucene104Codec(randomFrom(Zstd814StoredFieldsFormat.Mode.values())) {
            @Override
            public KnnVectorsFormat getKnnVectorsFormatForField(String field) {
                return new Lucene99HnswVectorsFormat();
            }
        });
        iwc.setMergePolicy(NoMergePolicy.INSTANCE);
        try (IndexWriter w = new IndexWriter(d, iwc)) {
            for (int i = 0; i < numDocs; i++) {
                Document document = new Document();
                document.add(new KnnFloatVectorField(FIELD_NAME, randomFloatVector(numDims), VectorSimilarityFunction.COSINE));
                w.addDocument(document);
                if (randomBoolean() && (i % 10 == 0)) {
                    w.commit();
                }
            }
            w.commit();
        }
    }

    private record Range(long offset, long length) {}

    private record BulkCall(long[] offsets, int length) {}

    /**
     * Records the prefetch hints received by the inputs of a {@link RecordingDirectory}, and scripts the outcomes of
     * {@link TieredPrefetchInput} calls. When {@code script} is null the inputs do not implement {@link TieredPrefetchInput},
     * which models an ordinary local directory.
     */
    private static final class PrefetchRecorder {
        private final Function<Long, TieredPrefetchInput.Outcome> script;
        private final List<BulkCall> bulkCalls = Collections.synchronizedList(new ArrayList<>());
        private final List<Range> prefetches = Collections.synchronizedList(new ArrayList<>());
        private final AtomicInteger singleEnsureResidentCalls = new AtomicInteger();

        PrefetchRecorder(Function<Long, TieredPrefetchInput.Outcome> script) {
            this.script = script;
        }

        void reset() {
            bulkCalls.clear();
            prefetches.clear();
            singleEnsureResidentCalls.set(0);
        }

        IndexInput wrap(String description, IndexInput in) {
            return script == null ? new RecordingIndexInput(description, in, this) : new TieredRecordingIndexInput(description, in, this);
        }
    }

    /**
     * A test double for a directory whose files live in a tiered store. The only production {@link TieredPrefetchInput}
     * lives in the stateless plugin, which server tests cannot depend on, so this wraps a real directory and makes its
     * inputs (including slices and clones, which is where vector values actually read from) record prefetch hints and,
     * optionally, implement {@link TieredPrefetchInput} with scripted outcomes. Reads always go to the real input.
     */
    private static final class RecordingDirectory extends FilterDirectory {
        private final PrefetchRecorder recorder;

        RecordingDirectory(Directory in, PrefetchRecorder recorder) {
            super(in);
            this.recorder = recorder;
        }

        @Override
        public IndexInput openInput(String name, IOContext context) throws IOException {
            return recorder.wrap(name, super.openInput(name, context));
        }
    }

    /**
     * Delegates all reads to a real input and records {@link IndexInput#prefetch} calls. Slices and clones are wrapped
     * too, each around the matching slice or clone of the delegate so file pointers stay independent.
     */
    private static class RecordingIndexInput extends IndexInput {
        protected final PrefetchRecorder recorder;
        private IndexInput in;

        RecordingIndexInput(String description, IndexInput in, PrefetchRecorder recorder) {
            super(description);
            this.in = in;
            this.recorder = recorder;
        }

        @Override
        public void close() throws IOException {
            in.close();
        }

        @Override
        public long getFilePointer() {
            return in.getFilePointer();
        }

        @Override
        public void seek(long pos) throws IOException {
            in.seek(pos);
        }

        @Override
        public long length() {
            return in.length();
        }

        @Override
        public IndexInput slice(String sliceDescription, long offset, long length) throws IOException {
            return recorder.wrap(sliceDescription, in.slice(sliceDescription, offset, length));
        }

        @Override
        public RecordingIndexInput clone() {
            RecordingIndexInput clone = (RecordingIndexInput) super.clone();
            clone.in = in.clone();
            return clone;
        }

        @Override
        public byte readByte() throws IOException {
            return in.readByte();
        }

        @Override
        public void readBytes(byte[] b, int offset, int len) throws IOException {
            in.readBytes(b, offset, len);
        }

        @Override
        public short readShort() throws IOException {
            return in.readShort();
        }

        @Override
        public int readInt() throws IOException {
            return in.readInt();
        }

        @Override
        public long readLong() throws IOException {
            return in.readLong();
        }

        @Override
        public void readFloats(float[] floats, int offset, int len) throws IOException {
            in.readFloats(floats, offset, len);
        }

        @Override
        public void prefetch(long offset, long length) throws IOException {
            recorder.prefetches.add(new Range(offset, length));
            in.prefetch(offset, length);
        }
    }

    /**
     * A {@link RecordingIndexInput} that also implements {@link TieredPrefetchInput}, recording each call and answering
     * with the recorder's scripted outcome for each range's offset.
     */
    private static final class TieredRecordingIndexInput extends RecordingIndexInput implements TieredPrefetchInput {

        TieredRecordingIndexInput(String description, IndexInput in, PrefetchRecorder recorder) {
            super(description, in, recorder);
        }

        @Override
        public Outcome ensureResident(long offset, long length) {
            recorder.singleEnsureResidentCalls.incrementAndGet();
            return recorder.script.apply(offset);
        }

        @Override
        public void ensureResident(long[] offsets, int length, int count, Outcome[] outcomes) {
            recorder.bulkCalls.add(new BulkCall(Arrays.copyOf(offsets, count), length));
            if (TieredPrefetchInput.checkBulkArgs(offsets, length, count, outcomes)) {
                return;
            }
            for (int i = 0; i < count; i++) {
                assertThat(offsets[i] + length, lessThanOrEqualTo(length()));
                outcomes[i] = recorder.script.apply(offsets[i]);
            }
        }

        @Override
        public long residencyRegionSize() {
            return 1L << 16;
        }
    }

    /**
     * A mock query that is used to test profiling
     */
    private static class MockQueryProfilerProvider extends Query implements QueryProfilerProvider {

        private final long vectorOpsCount;

        private MockQueryProfilerProvider(long vectorOpsCount) {
            this.vectorOpsCount = vectorOpsCount;
        }

        @Override
        public String toString(String field) {
            return "";
        }

        @Override
        public Weight createWeight(IndexSearcher searcher, ScoreMode scoreMode, float boost) throws IOException {
            throw new UnsupportedEncodingException("Should have been rewritten");
        }

        @Override
        public Query rewrite(IndexSearcher indexSearcher) throws IOException {
            return Queries.ALL_DOCS_INSTANCE;
        }

        @Override
        public void visit(QueryVisitor visitor) {}

        @Override
        public boolean equals(Object obj) {
            return obj instanceof MockQueryProfilerProvider;
        }

        @Override
        public int hashCode() {
            return 0;
        }

        @Override
        public void profile(QueryProfiler queryProfiler) {
            queryProfiler.addVectorOpsCount(vectorOpsCount);
        }
    }

    private static void addRandomDocuments(int numDocs, Directory d, int numDims) throws IOException {
        IndexWriterConfig iwc = new IndexWriterConfig();
        // Pick codec from quantized vector formats to ensure scores use real scores when using knn rescore
        DenseVectorFieldMapper.ElementType elementType = randomFrom(
            DenseVectorFieldMapper.ElementType.FLOAT,
            DenseVectorFieldMapper.ElementType.BFLOAT16
        );
        KnnVectorsFormat format = randomFrom(
            /*new ES920DiskBBQVectorsFormat(
                DEFAULT_VECTORS_PER_CLUSTER,
                DEFAULT_CENTROIDS_PER_PARENT_CLUSTER,
                elementType,
                randomBoolean(),
                null,
                1
            ),
            new ES93BinaryQuantizedVectorsFormat(elementType, false),
            new ES93HnswBinaryQuantizedVectorsFormat(elementType, randomBoolean()),
            new ES93ScalarQuantizedVectorsFormat(elementType),*/
            new ES93HnswScalarQuantizedVectorsFormat(
                DEFAULT_VECTORS_PER_CLUSTER,
                DEFAULT_CENTROIDS_PER_PARENT_CLUSTER,
                elementType,
                null,
                7,
                false,
                randomBoolean()
            )
        );
        iwc.setCodec(new Elasticsearch93Lucene104Codec(randomFrom(Zstd814StoredFieldsFormat.Mode.values())) {
            @Override
            public KnnVectorsFormat getKnnVectorsFormatForField(String field) {
                return format;
            }
        });
        try (IndexWriter w = new IndexWriter(d, newIndexWriterConfig())) {
            for (int i = 0; i < numDocs; i++) {
                Document document = new Document();
                float[] vector = randomFloatVector(numDims);
                KnnFloatVectorField vectorField = new KnnFloatVectorField(FIELD_NAME, vector, VectorSimilarityFunction.COSINE);
                document.add(vectorField);
                w.addDocument(document);
                if (randomBoolean() && (i % 10 == 0)) {
                    w.commit();
                }
            }
            w.commit();
        }
    }

    private static class SingleVectorQueryIndexReader extends FilterDirectoryReader {

        /**
         * Create a new FilterDirectoryReader that filters a passed in DirectoryReader, using the supplied
         * SubReaderWrapper to wrap its subreader.
         *
         * @param in      the DirectoryReader to filter
         */
        SingleVectorQueryIndexReader(DirectoryReader in) throws IOException {
            super(in, new SubReaderWrapper() {
                @Override
                public LeafReader wrap(LeafReader reader) {
                    return new FilterLeafReader(reader) {
                        @Override
                        public CacheHelper getReaderCacheHelper() {
                            return null;
                        }

                        @Override
                        public CacheHelper getCoreCacheHelper() {
                            return null;
                        }

                        @Override
                        public FloatVectorValues getFloatVectorValues(String field) throws IOException {
                            FloatVectorValues values = super.getFloatVectorValues(field);
                            if (values == null) {
                                return null;
                            }
                            return new SingleFloatVectorValues(values);
                        }
                    };
                }
            });
        }

        @Override
        protected DirectoryReader doWrapDirectoryReader(DirectoryReader in) throws IOException {
            return new SingleVectorQueryIndexReader(in);
        }

        @Override
        public CacheHelper getReaderCacheHelper() {
            return null;
        }
    }

    /**
     * A wrapper around FloatVectorValues that ensures that the bulk scoring path uses the single scoring method.
     * Used to test that the single and bulk scoring paths return the same scores.
     */
    private static final class SingleFloatVectorValues extends FloatVectorValues {

        private final FloatVectorValues in;

        SingleFloatVectorValues(FloatVectorValues in) {
            this.in = in;
        }

        @Override
        public VectorScorer scorer(float[] target) throws IOException {
            return new SingleVectorScorer(in.scorer(target));
        }

        @Override
        public VectorScorer rescorer(float[] target) throws IOException {
            return new SingleVectorScorer(in.rescorer(target));
        }

        @Override
        public int ordToDoc(int ord) {
            return in.ordToDoc(ord);
        }

        @Override
        public Bits getAcceptOrds(Bits acceptDocs) {
            return in.getAcceptOrds(acceptDocs);
        }

        @Override
        public DocIndexIterator iterator() {
            return in.iterator();
        }

        @Override
        public int getVectorByteLength() {
            return in.getVectorByteLength();
        }

        @Override
        public float[] vectorValue(int ord) throws IOException {
            return in.vectorValue(ord);
        }

        @Override
        public FloatVectorValues copy() throws IOException {
            return new SingleFloatVectorValues(in.copy());
        }

        @Override
        public int dimension() {
            return in.dimension();
        }

        @Override
        public int size() {
            return in.size();
        }
    }

    private static final class SingleVectorScorer implements VectorScorer {
        private final VectorScorer in;

        SingleVectorScorer(VectorScorer in) {
            this.in = in;
        }

        @Override
        public float score() throws IOException {
            return in.score();
        }

        @Override
        public DocIdSetIterator iterator() {
            return in.iterator();
        }

        @Override
        public VectorScorer.Bulk bulk(DocIdSetIterator matchingDocs) throws IOException {
            final DocIdSetIterator iterator = matchingDocs == null
                ? iterator()
                : ConjunctionUtils.createConjunction(List.of(matchingDocs, iterator()), List.of());
            if (iterator.docID() == -1) {
                iterator.nextDoc();
            }
            return (upTo, liveDocs, buffer) -> {
                assert upTo > 0;
                buffer.growNoCopy(VectorScorer.DEFAULT_BULK_BATCH_SIZE);
                int size = 0;
                float maxScore = Float.NEGATIVE_INFINITY;
                for (int doc = iterator.docID(); doc < upTo && size < VectorScorer.DEFAULT_BULK_BATCH_SIZE; doc = iterator.nextDoc()) {
                    if (liveDocs == null || liveDocs.get(doc)) {
                        buffer.docs[size] = doc;
                        buffer.features[size] = score();
                        maxScore = Math.max(maxScore, buffer.features[size]);
                        ++size;
                    }
                }
                buffer.size = size;
                return maxScore;
            };
        }
    }
}
