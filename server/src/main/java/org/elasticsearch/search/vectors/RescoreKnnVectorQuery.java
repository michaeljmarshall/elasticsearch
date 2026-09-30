/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the "Elastic License
 * 2.0", the "GNU Affero General Public License v3.0 only", and the "Server Side
 * Public License v 1"; you may not use this file except in compliance with, at
 * your election, the "Elastic License 2.0", the "GNU Affero General Public
 * License v3.0 only", or the "Server Side Public License, v 1".
 */

package org.elasticsearch.search.vectors;

import org.apache.lucene.codecs.lucene95.HasIndexSlice;
import org.apache.lucene.index.ByteVectorValues;
import org.apache.lucene.index.FloatVectorValues;
import org.apache.lucene.index.KnnVectorValues;
import org.apache.lucene.queries.function.FunctionScoreQuery;
import org.apache.lucene.search.BooleanClause;
import org.apache.lucene.search.ConjunctionUtils;
import org.apache.lucene.search.DocAndFloatFeatureBuffer;
import org.apache.lucene.search.DocIdSetIterator;
import org.apache.lucene.search.IndexSearcher;
import org.apache.lucene.search.KnnByteVectorQuery;
import org.apache.lucene.search.KnnFloatVectorQuery;
import org.apache.lucene.search.MatchAllDocsQuery;
import org.apache.lucene.search.MatchNoDocsQuery;
import org.apache.lucene.search.Query;
import org.apache.lucene.search.QueryVisitor;
import org.apache.lucene.search.ScoreDoc;
import org.apache.lucene.search.ScoreMode;
import org.apache.lucene.search.TopDocs;
import org.apache.lucene.search.VectorScorer;
import org.apache.lucene.store.IndexInput;
import org.apache.lucene.util.ArrayUtil;
import org.elasticsearch.common.lucene.search.Queries;
import org.elasticsearch.core.TieredPrefetchInput;
import org.elasticsearch.search.profile.query.QueryProfiler;
import org.elasticsearch.search.profile.query.TieredPrefetchOutcomeCounts;

import java.io.IOException;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Objects;

/**
 * A Lucene {@link Query} that applies vector-based rescoring to an inner query's results.
 * <p>
 * Depending on the nature of the {@code innerQuery}, this class dynamically selects between two rescoring strategies:
 * <ul>
 *   <li><b>Inline rescoring</b>:
 *   Used when the inner query is already a top-N vector query with {@code rescoreK} results.
 *   The vector similarity is applied inline using a {@link FunctionScoreQuery} without an additional
 *   filtering pass.</li>
 *   <li><b>Late rescoring</b>: Used when the inner query is not a top-N vector query or does not return
 *   {@code rescoreK} results. The top {@code rescoreK} documents are first collected, and then rescoring is applied
 *   separately to select the final top {@code k}.</li>
 * </ul>
 */
public abstract class RescoreKnnVectorQuery extends Query implements QueryProfilerProvider {

    /**
     * A sealed interface representing a query vector target for kNN search operations.
     * Eliminates null-based dispatch between float[] and byte[] query vectors.
     */
    public sealed interface VectorQueryTarget permits VectorQueryTarget.FloatTarget, VectorQueryTarget.ByteTarget {

        /** Returns the dimensionality of the query vector. */
        int dimension();

        /** A float[] query vector target. */
        record FloatTarget(float[] vector) implements VectorQueryTarget {
            @Override
            public int dimension() {
                return vector.length;
            }

            @Override
            public boolean equals(Object o) {
                return o instanceof FloatTarget ft && Arrays.equals(vector, ft.vector);
            }

            @Override
            public int hashCode() {
                return Arrays.hashCode(vector);
            }

            @Override
            public String toString() {
                return "floatTarget=" + vector[0] + "...";
            }
        }

        /** A byte[] query vector target. */
        record ByteTarget(byte[] vector) implements VectorQueryTarget {
            @Override
            public int dimension() {
                return vector.length;
            }

            @Override
            public boolean equals(Object o) {
                return o instanceof ByteTarget bt && Arrays.equals(vector, bt.vector);
            }

            @Override
            public int hashCode() {
                return Arrays.hashCode(vector);
            }

            @Override
            public String toString() {
                return "byteTarget=" + vector[0] + "...";
            }
        }
    }

    protected final String fieldName;
    protected final VectorQueryTarget target;
    protected final int k;
    protected final Query innerQuery;
    protected long vectorOperations = 0;
    /**
     * Outcomes of the tiered prefetch requests issued while rescoring. Replaced on each rewrite, like
     * {@link #vectorOperations} is overwritten on each rewrite.
     */
    protected TieredPrefetchOutcomeCounts tieredPrefetchOutcomes = new TieredPrefetchOutcomeCounts();

    private RescoreKnnVectorQuery(String fieldName, VectorQueryTarget target, int k, Query innerQuery) {
        this.fieldName = fieldName;
        this.target = target;
        this.k = k;
        this.innerQuery = innerQuery;
    }

    /**
     * Selects and returns the appropriate {@link RescoreKnnVectorQuery} strategy based on the nature of the {@code innerQuery}.
     *
     * @param fieldName                the name of the field containing the vector
     * @param floatTarget              the target vector to compare against
     * @param k                        the number of top documents to return after rescoring
     * @param rescoreK                 the number of top documents to consider for rescoring
     * @param innerQuery               the original Lucene query to rescore
     */
    public static RescoreKnnVectorQuery fromInnerQuery(String fieldName, float[] floatTarget, int k, int rescoreK, Query innerQuery) {
        return fromInnerQuery(fieldName, new VectorQueryTarget.FloatTarget(floatTarget), k, rescoreK, innerQuery);
    }

    /**
     * Selects and returns the appropriate {@link RescoreKnnVectorQuery} strategy for byte vector fields.
     *
     * @param fieldName                the name of the field containing the vector
     * @param byteTarget               the byte target vector to compare against
     * @param k                        the number of top documents to return after rescoring
     * @param rescoreK                 the number of top documents to consider for rescoring
     * @param innerQuery               the original Lucene query to rescore
     */
    public static RescoreKnnVectorQuery fromInnerQuery(String fieldName, byte[] byteTarget, int k, int rescoreK, Query innerQuery) {
        return fromInnerQuery(fieldName, new VectorQueryTarget.ByteTarget(byteTarget), k, rescoreK, innerQuery);
    }

    private static RescoreKnnVectorQuery fromInnerQuery(String fieldName, VectorQueryTarget target, int k, int rescoreK, Query innerQuery) {
        if ((innerQuery instanceof KnnFloatVectorQuery fQuery && fQuery.getK() == rescoreK)
            || (innerQuery instanceof KnnByteVectorQuery bQuery && bQuery.getK() == rescoreK)) {
            // Queries that return only the top `k` results and do not require reduction before re-scoring.
            return new InlineRescoreQuery(fieldName, target, k, innerQuery);
        }
        return new LateRescoreQuery(fieldName, target, k, rescoreK, innerQuery);
    }

    public Query innerQuery() {
        return innerQuery;
    }

    public int k() {
        return k;
    }

    /**
     * The {@link TieredPrefetchInput} outcomes observed by the last rewrite of this query. All counts are zero when the
     * vector data is not backed by an input implementing {@link TieredPrefetchInput}.
     */
    public TieredPrefetchOutcomeCounts tieredPrefetchOutcomes() {
        return tieredPrefetchOutcomes;
    }

    @Override
    public void profile(QueryProfiler queryProfiler) {
        if (innerQuery instanceof QueryProfilerProvider queryProfilerProvider) {
            queryProfilerProvider.profile(queryProfiler);
        }

        queryProfiler.addVectorOpsCount(vectorOperations);
        queryProfiler.addTieredPrefetchOutcomes(tieredPrefetchOutcomes);
    }

    @Override
    public void visit(QueryVisitor visitor) {
        innerQuery.visit(visitor.getSubVisitor(BooleanClause.Occur.MUST, this));
    }

    @Override
    public boolean equals(Object o) {
        if (this == o) return true;
        if (o == null || getClass() != o.getClass()) return false;
        RescoreKnnVectorQuery that = (RescoreKnnVectorQuery) o;
        return Objects.equals(fieldName, that.fieldName)
            && Objects.equals(target, that.target)
            && Objects.equals(k, that.k)
            && Objects.equals(innerQuery, that.innerQuery);
    }

    @Override
    public int hashCode() {
        return Objects.hash(fieldName, target, k, innerQuery);
    }

    @Override
    public String toString(String field) {
        return getClass().getSimpleName()
            + "{"
            + "fieldName='"
            + fieldName
            + '\''
            + ", "
            + target
            + ", k="
            + k
            + ", vectorQuery="
            + innerQuery
            + '}';
    }

    private static class InlineRescoreQuery extends RescoreKnnVectorQuery {
        private InlineRescoreQuery(String fieldName, VectorQueryTarget target, int k, Query innerQuery) {
            super(fieldName, target, k, innerQuery);
        }

        @Override
        public Query rewrite(IndexSearcher searcher) throws IOException {
            tieredPrefetchOutcomes = new TieredPrefetchOutcomeCounts();
            var rescoreQuery = new DirectRescoreKnnVectorQuery(fieldName, target, innerQuery, tieredPrefetchOutcomes);
            var topDocs = searcher.search(rescoreQuery, k);
            vectorOperations = topDocs.totalHits.value();
            return new KnnScoreDocQuery(topDocs.scoreDocs, searcher.getIndexReader());
        }

        @Override
        public boolean equals(Object o) {
            if (this == o) return true;
            if (o == null || getClass() != o.getClass()) return false;
            return super.equals(o);
        }

        @Override
        public int hashCode() {
            return super.hashCode();
        }
    }

    private static class LateRescoreQuery extends RescoreKnnVectorQuery {
        final int rescoreK;

        private LateRescoreQuery(String fieldName, VectorQueryTarget target, int k, int rescoreK, Query innerQuery) {
            super(fieldName, target, k, innerQuery);
            this.rescoreK = rescoreK;
        }

        @Override
        public Query rewrite(IndexSearcher searcher) throws IOException {
            final TopDocs topDocs;
            // Retrieve top `rescoreK` documents from the inner query
            topDocs = searcher.search(innerQuery, rescoreK);
            vectorOperations = topDocs.totalHits.value();

            // Retrieve top `k` documents from the top `rescoreK` query
            var topDocsQuery = new KnnScoreDocQuery(topDocs.scoreDocs, searcher.getIndexReader());
            tieredPrefetchOutcomes = new TieredPrefetchOutcomeCounts();
            var rescoreQuery = new DirectRescoreKnnVectorQuery(fieldName, target, topDocsQuery, tieredPrefetchOutcomes);
            var rescoreTopDocs = searcher.search(rescoreQuery.rewrite(searcher), k);
            return new KnnScoreDocQuery(rescoreTopDocs.scoreDocs, searcher.getIndexReader());
        }

        @Override
        public boolean equals(Object o) {
            if (this == o) return true;
            if (o == null || getClass() != o.getClass()) return false;
            var that = (RescoreKnnVectorQuery.LateRescoreQuery) o;
            return super.equals(o) && that.rescoreK == rescoreK;
        }

        @Override
        public int hashCode() {
            return Objects.hash(super.hashCode(), rescoreK);
        }
    }

    private static class DirectRescoreKnnVectorQuery extends Query {
        private static final int BULK_SCORE_SIZE = 32;

        private final VectorQueryTarget target;
        private final String fieldName;
        private final Query innerQuery;
        private final TieredPrefetchOutcomeCounts tieredPrefetchOutcomes;

        DirectRescoreKnnVectorQuery(
            String fieldName,
            VectorQueryTarget target,
            Query innerQuery,
            TieredPrefetchOutcomeCounts tieredPrefetchOutcomes
        ) {
            this.fieldName = fieldName;
            this.target = target;
            this.innerQuery = innerQuery;
            this.tieredPrefetchOutcomes = tieredPrefetchOutcomes;
        }

        @Override
        public String toString(String field) {
            return "DirectRescoreKnnVectorQuery[" + innerQuery + "]";
        }

        /**
         * Iterates over the first {@code count} entries of a doc-ordered array of one leaf's candidate doc IDs.
         * It starts positioned on the first candidate, so the bulk scorer can consume it without an initial
         * {@code nextDoc()}; the vector scorer's own iterator is advanced to the same doc before the bulk scorer is created.
         */
        static class CandidateIterator extends DocIdSetIterator {

            private final int[] docIds;
            private final int count;
            private int idx = 0;    // just start on the first candidate without needing an initial advance

            CandidateIterator(int[] docIds, int count) {
                assert count > 0;
                this.docIds = docIds;
                this.count = count;
            }

            @Override
            public int docID() {
                if (idx == NO_MORE_DOCS) {
                    return idx;
                }
                return docIds[idx];
            }

            @Override
            public int nextDoc() {
                idx++;
                if (idx >= count) {
                    idx = NO_MORE_DOCS;
                }
                return docID();
            }

            @Override
            public int advance(int target) throws IOException {
                return slowAdvance(target);
            }

            @Override
            public long cost() {
                return count;
            }
        }

        /**
         * Rescores leaf by leaf in two passes. The first pass materializes every candidate (doc, ord) of the leaf; the
         * candidate set is bounded by {@code rescoreK}, so this is small. Materializing first lets us hand all of a
         * leaf's candidate vectors to a {@link TieredPrefetchInput} in one bulk call, so a tiered store can start every
         * remote fetch the leaf needs at once instead of discovering misses one blocking read at a time. Inputs that are
         * not tiered get the plain {@link IndexInput#prefetch} hint per candidate, as before. The second pass scores the
         * candidates in doc order with one forward-only bulk scorer per leaf. Outcomes never change which docs are
         * scored or in what order; they are only counted.
         */
        @Override
        public Query rewrite(IndexSearcher indexSearcher) throws IOException {
            Query innerRewritten = innerQuery.rewrite(indexSearcher);
            if (innerRewritten.getClass() == MatchNoDocsQuery.class) {
                return Queries.NO_DOCS_INSTANCE;
            }
            assert innerRewritten.getClass() != MatchAllDocsQuery.class;

            DocAndFloatFeatureBuffer buffer = new DocAndFloatFeatureBuffer();
            List<ScoreDoc> results = new ArrayList<>(10);

            // candidate buffers, reused across leaves
            int[] docs = new int[0];
            int[] ords = new int[0];
            long[] offsets = new long[0];
            TieredPrefetchInput.Outcome[] outcomes = new TieredPrefetchInput.Outcome[0];

            for (var leaf : indexSearcher.getIndexReader().leaves()) {
                var fieldInfo = leaf.reader().getFieldInfos().fieldInfo(fieldName);
                if (fieldInfo == null) {
                    continue;
                }
                KnnVectorValues knnVectorValues;
                if (target instanceof VectorQueryTarget.ByteTarget) {
                    knnVectorValues = leaf.reader().getByteVectorValues(fieldName);
                } else {
                    knnVectorValues = leaf.reader().getFloatVectorValues(fieldName);
                }
                if (knnVectorValues == null) {
                    continue;
                }
                if (knnVectorValues.dimension() != target.dimension()) {
                    throw new IllegalArgumentException(
                        "vector query dimension: " + target.dimension() + " differs from field dimension: " + knnVectorValues.dimension()
                    );
                }
                var weight = innerRewritten.createWeight(indexSearcher, ScoreMode.COMPLETE_NO_SCORES, 1.0f);
                var scorer = weight.scorer(leaf);
                if (scorer == null) {
                    continue;
                }
                var filterIterator = scorer.iterator();

                final int vectorByteSize = knnVectorValues.getVectorByteLength();
                final IndexInput input = getIndexSliceOrNull(knnVectorValues);
                // slices of a tiered input may be plain heap inputs, so check the slice itself rather than the file
                final TieredPrefetchInput tieredInput = input instanceof TieredPrefetchInput t ? t : null;
                KnnVectorValues.DocIndexIterator vectorIter = knnVectorValues.iterator();
                DocIdSetIterator conjunction = ConjunctionUtils.intersectIterators(List.of(vectorIter, filterIterator));

                // pass 1: materialize the leaf's candidates in doc order
                int count = 0;
                int doc;
                while ((doc = conjunction.nextDoc()) != DocIdSetIterator.NO_MORE_DOCS) {
                    assert doc == vectorIter.docID();
                    final int ord = vectorIter.index();

                    if (tieredInput == null && input != null) {
                        input.prefetch((long) ord * vectorByteSize, vectorByteSize);
                    }

                    if (count == docs.length) {
                        docs = ArrayUtil.grow(docs, count + 1);
                        ords = ArrayUtil.grow(ords, count + 1);
                    }
                    docs[count] = doc;
                    ords[count] = ord;
                    count++;
                }
                if (count == 0) {
                    continue;
                }

                if (tieredInput != null) {
                    if (offsets.length < count) {
                        offsets = new long[ArrayUtil.oversize(count, Long.BYTES)];
                        outcomes = new TieredPrefetchInput.Outcome[offsets.length];
                    }
                    for (int i = 0; i < count; i++) {
                        offsets[i] = (long) ords[i] * vectorByteSize;
                    }
                    // one bulk call per leaf; the input groups the ranges by region internally
                    tieredInput.ensureResident(offsets, vectorByteSize, count, outcomes);
                    for (int i = 0; i < count; i++) {
                        tieredPrefetchOutcomes.add(outcomes[i]);
                    }
                    // Reordering leaves or candidates by outcome (e.g. resident first) is deliberately deferred until measured.
                }

                // pass 2: score the candidates in doc order
                VectorScorer vecScorer = switch (target) {
                    case VectorQueryTarget.ByteTarget bt -> ((ByteVectorValues) knnVectorValues).rescorer(bt.vector());
                    case VectorQueryTarget.FloatTarget ft -> ((FloatVectorValues) knnVectorValues).rescorer(ft.vector());
                };
                scoreLeaf(vecScorer, docs, count, leaf.docBase, buffer, results);
            }

            return new KnnScoreDocQuery(results.toArray(ScoreDoc[]::new), indexSearcher.getIndexReader());
        }

        private static IndexInput getIndexSliceOrNull(KnnVectorValues vectorValues) {
            return vectorValues instanceof HasIndexSlice h ? h.getSlice() : null;
        }

        /**
         * Scores the first {@code count} doc-ordered candidates of one leaf in batches of {@link #BULK_SCORE_SIZE}.
         * Only one bulk scorer can be created per {@link VectorScorer}, and it is forward-only, so a single bulk scorer
         * covers the whole leaf and is advanced batch by batch.
         */
        private static void scoreLeaf(
            VectorScorer scorer,
            int[] docs,
            int count,
            int docBase,
            DocAndFloatFeatureBuffer buffer,
            List<ScoreDoc> results
        ) throws IOException {
            CandidateIterator iterator = new CandidateIterator(docs, count);
            scorer.iterator().advance(docs[0]);
            VectorScorer.Bulk bulkScorer = scorer.bulk(iterator);

            for (int start = 0; start < count; start += BULK_SCORE_SIZE) {
                int batchSize = Math.min(BULK_SCORE_SIZE, count - start);
                int maxDocId = docs[start + batchSize - 1] + 1; // upTo is EXCLUSIVE
                bulkScorer.nextDocsAndScores(maxDocId, null, buffer);
                assert buffer.size == batchSize;

                for (int d = 0; d < buffer.size; d++) {
                    if (Float.isNaN(buffer.features[d]) == false) {
                        results.add(new ScoreDoc(buffer.docs[d] + docBase, buffer.features[d]));
                    }
                }
            }
        }

        @Override
        public void visit(QueryVisitor visitor) {
            if (visitor.acceptField(fieldName)) {
                visitor.visitLeaf(this);
            }
        }

        @Override
        public boolean equals(Object obj) {
            if (this == obj) return true;
            if (obj == null || getClass() != obj.getClass()) return false;
            DirectRescoreKnnVectorQuery that = (DirectRescoreKnnVectorQuery) obj;
            return Objects.equals(innerQuery, that.innerQuery);
        }

        @Override
        public int hashCode() {
            return Objects.hash(innerQuery, getClass());
        }
    }
}
