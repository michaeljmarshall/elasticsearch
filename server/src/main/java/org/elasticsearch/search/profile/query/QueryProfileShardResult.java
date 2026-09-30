/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the "Elastic License
 * 2.0", the "GNU Affero General Public License v3.0 only", and the "Server Side
 * Public License v 1"; you may not use this file except in compliance with, at
 * your election, the "Elastic License 2.0", the "GNU Affero General Public
 * License v3.0 only", or the "Server Side Public License, v 1".
 */

package org.elasticsearch.search.profile.query;

import org.elasticsearch.TransportVersion;
import org.elasticsearch.common.Strings;
import org.elasticsearch.common.io.stream.StreamInput;
import org.elasticsearch.common.io.stream.StreamOutput;
import org.elasticsearch.common.io.stream.Writeable;
import org.elasticsearch.core.Nullable;
import org.elasticsearch.search.profile.ProfileResult;
import org.elasticsearch.xcontent.ToXContentObject;
import org.elasticsearch.xcontent.XContentBuilder;

import java.io.IOException;
import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.util.Objects;

/**
 * A container class to hold the profile results for a single shard in the request.
 * Contains a list of query profiles, a collector tree and a total rewrite tree.
 */
public final class QueryProfileShardResult implements Writeable, ToXContentObject {

    private static final TransportVersion QUERY_PROFILE_TIERED_PREFETCH_OUTCOMES = TransportVersion.fromName(
        "query_profile_tiered_prefetch_outcomes"
    );

    public static final String COLLECTOR = "collector";
    public static final String REWRITE_TIME = "rewrite_time";
    public static final String QUERY_ARRAY = "query";

    public static final String VECTOR_OPERATIONS_COUNT = "vector_operations_count";
    public static final String TIERED_PREFETCH = "tiered_prefetch";

    private final List<ProfileResult> queryProfileResults;

    private final CollectorResult profileCollector;

    private final long rewriteTime;

    private final Long vectorOperationsCount;

    /**
     * How the query's prefetch requests against a tiered store came out, or {@code null} when the query did not read
     * through a tiered input. Absent from the response in that case, so output on nodes without tiered storage is
     * unchanged.
     */
    private final TieredPrefetchOutcomeCounts tieredPrefetchOutcomes;

    public QueryProfileShardResult(
        List<ProfileResult> queryProfileResults,
        long rewriteTime,
        CollectorResult profileCollector,
        @Nullable Long vectorOperationsCount
    ) {
        this(queryProfileResults, rewriteTime, profileCollector, vectorOperationsCount, null);
    }

    public QueryProfileShardResult(
        List<ProfileResult> queryProfileResults,
        long rewriteTime,
        CollectorResult profileCollector,
        @Nullable Long vectorOperationsCount,
        @Nullable TieredPrefetchOutcomeCounts tieredPrefetchOutcomes
    ) {
        assert (profileCollector != null);
        this.queryProfileResults = queryProfileResults;
        this.profileCollector = profileCollector;
        this.rewriteTime = rewriteTime;
        this.vectorOperationsCount = vectorOperationsCount;
        this.tieredPrefetchOutcomes = tieredPrefetchOutcomes;
    }

    /**
     * Read from a stream.
     */
    public QueryProfileShardResult(StreamInput in) throws IOException {
        int profileSize = in.readVInt();
        queryProfileResults = new ArrayList<>(profileSize);
        for (int j = 0; j < profileSize; j++) {
            queryProfileResults.add(new ProfileResult(in));
        }

        profileCollector = new CollectorResult(in);
        rewriteTime = in.readLong();
        vectorOperationsCount = in.readOptionalLong();
        if (in.getTransportVersion().supports(QUERY_PROFILE_TIERED_PREFETCH_OUTCOMES)) {
            tieredPrefetchOutcomes = in.readOptionalWriteable(TieredPrefetchOutcomeCounts::new);
        } else {
            tieredPrefetchOutcomes = null;
        }
    }

    @Override
    public void writeTo(StreamOutput out) throws IOException {
        out.writeVInt(queryProfileResults.size());
        for (ProfileResult p : queryProfileResults) {
            p.writeTo(out);
        }
        profileCollector.writeTo(out);
        out.writeLong(rewriteTime);
        out.writeOptionalLong(vectorOperationsCount);
        if (out.getTransportVersion().supports(QUERY_PROFILE_TIERED_PREFETCH_OUTCOMES)) {
            out.writeOptionalWriteable(tieredPrefetchOutcomes);
        }
    }

    public List<ProfileResult> getQueryResults() {
        return Collections.unmodifiableList(queryProfileResults);
    }

    public long getRewriteTime() {
        return rewriteTime;
    }

    public CollectorResult getCollectorResult() {
        return profileCollector;
    }

    @Override
    public XContentBuilder toXContent(XContentBuilder builder, Params params) throws IOException {
        builder.startObject();
        if (vectorOperationsCount != null) {
            builder.field(VECTOR_OPERATIONS_COUNT, vectorOperationsCount);
        }
        if (tieredPrefetchOutcomes != null) {
            builder.field(TIERED_PREFETCH);
            tieredPrefetchOutcomes.toXContent(builder, params);
        }
        builder.startArray(QUERY_ARRAY);
        for (ProfileResult p : queryProfileResults) {
            p.toXContent(builder, params);
        }
        builder.endArray();
        builder.field(REWRITE_TIME, rewriteTime);
        builder.startArray(COLLECTOR);
        profileCollector.toXContent(builder, params);
        builder.endArray();
        builder.endObject();
        return builder;
    }

    @Override
    public boolean equals(Object obj) {
        if (obj == null || getClass() != obj.getClass()) {
            return false;
        }
        QueryProfileShardResult other = (QueryProfileShardResult) obj;
        return queryProfileResults.equals(other.queryProfileResults)
            && profileCollector.equals(other.profileCollector)
            && rewriteTime == other.rewriteTime
            && Objects.equals(tieredPrefetchOutcomes, other.tieredPrefetchOutcomes);
    }

    @Override
    public int hashCode() {
        return Objects.hash(queryProfileResults, profileCollector, rewriteTime, tieredPrefetchOutcomes);
    }

    @Override
    public String toString() {
        return Strings.toString(this);
    }

    public Long getVectorOperationsCount() {
        return vectorOperationsCount;
    }

    @Nullable
    public TieredPrefetchOutcomeCounts getTieredPrefetchOutcomes() {
        return tieredPrefetchOutcomes;
    }
}
