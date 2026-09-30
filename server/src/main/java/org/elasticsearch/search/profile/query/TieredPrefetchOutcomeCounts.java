/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the "Elastic License
 * 2.0", the "GNU Affero General Public License v3.0 only", and the "Server Side
 * Public License v 1"; you may not use this file except in compliance with, at
 * your election, the "Elastic License 2.0", the "GNU Affero General Public
 * License v3.0 only", or the "Server Side Public License, v 1".
 */

package org.elasticsearch.search.profile.query;

import org.elasticsearch.common.io.stream.StreamInput;
import org.elasticsearch.common.io.stream.StreamOutput;
import org.elasticsearch.common.io.stream.Writeable;
import org.elasticsearch.core.TieredPrefetchInput;
import org.elasticsearch.xcontent.ToXContentObject;
import org.elasticsearch.xcontent.XContentBuilder;
import org.elasticsearch.xcontent.XContentParser;

import java.io.IOException;

import static org.elasticsearch.common.xcontent.XContentParserUtils.ensureExpectedToken;

/**
 * Tallies the {@link TieredPrefetchInput.Outcome}s of prefetch requests made by a query. The split between resident,
 * fetching and skipped ranges shows how much of a query's data had to come from the remote tier of a tiered store,
 * which is what decides whether deeper or earlier prefetching is worth it. It is reported in the search profile so the
 * split can be read per shard, next to {@code vector_operations_count}.
 *
 * <p>Not thread-safe: a query accumulates into its own instance on the thread that rewrites it, and the profiler takes
 * a {@link #copy()} when it builds its result.
 */
public final class TieredPrefetchOutcomeCounts implements Writeable, ToXContentObject {

    public static final String RESIDENT = "resident";
    public static final String FETCHING = "fetching";
    public static final String SKIPPED = "skipped";

    private long resident;
    private long fetching;
    private long skipped;

    public TieredPrefetchOutcomeCounts() {}

    public TieredPrefetchOutcomeCounts(long resident, long fetching, long skipped) {
        this.resident = resident;
        this.fetching = fetching;
        this.skipped = skipped;
    }

    public TieredPrefetchOutcomeCounts(StreamInput in) throws IOException {
        this(in.readVLong(), in.readVLong(), in.readVLong());
    }

    @Override
    public void writeTo(StreamOutput out) throws IOException {
        out.writeVLong(resident);
        out.writeVLong(fetching);
        out.writeVLong(skipped);
    }

    /**
     * Records one outcome.
     */
    public void add(TieredPrefetchInput.Outcome outcome) {
        switch (outcome) {
            case RESIDENT -> resident++;
            case FETCHING -> fetching++;
            case SKIPPED -> skipped++;
        }
    }

    /**
     * Adds all counts of {@code other} to this instance.
     */
    public void add(TieredPrefetchOutcomeCounts other) {
        resident += other.resident;
        fetching += other.fetching;
        skipped += other.skipped;
    }

    /** A snapshot that no longer changes when this instance accumulates further. */
    public TieredPrefetchOutcomeCounts copy() {
        return new TieredPrefetchOutcomeCounts(resident, fetching, skipped);
    }

    public long resident() {
        return resident;
    }

    public long fetching() {
        return fetching;
    }

    public long skipped() {
        return skipped;
    }

    public long total() {
        return resident + fetching + skipped;
    }

    /** {@code true} when no outcome was recorded, i.e. the query did not read through a tiered input. */
    public boolean isEmpty() {
        return total() == 0;
    }

    @Override
    public XContentBuilder toXContent(XContentBuilder builder, Params params) throws IOException {
        builder.startObject();
        builder.field(RESIDENT, resident);
        builder.field(FETCHING, fetching);
        builder.field(SKIPPED, skipped);
        builder.endObject();
        return builder;
    }

    /**
     * Parses the object written by {@link #toXContent}. Unknown fields are skipped so that a newer node's output can be
     * read by an older client.
     */
    public static TieredPrefetchOutcomeCounts fromXContent(XContentParser parser) throws IOException {
        ensureExpectedToken(XContentParser.Token.START_OBJECT, parser.currentToken(), parser);
        long resident = 0;
        long fetching = 0;
        long skipped = 0;
        String currentFieldName = null;
        XContentParser.Token token;
        while ((token = parser.nextToken()) != XContentParser.Token.END_OBJECT) {
            if (token == XContentParser.Token.FIELD_NAME) {
                currentFieldName = parser.currentName();
            } else if (token.isValue()) {
                if (RESIDENT.equals(currentFieldName)) {
                    resident = parser.longValue();
                } else if (FETCHING.equals(currentFieldName)) {
                    fetching = parser.longValue();
                } else if (SKIPPED.equals(currentFieldName)) {
                    skipped = parser.longValue();
                }
            } else {
                parser.skipChildren();
            }
        }
        return new TieredPrefetchOutcomeCounts(resident, fetching, skipped);
    }

    @Override
    public boolean equals(Object o) {
        if (this == o) {
            return true;
        }
        if (o == null || getClass() != o.getClass()) {
            return false;
        }
        TieredPrefetchOutcomeCounts that = (TieredPrefetchOutcomeCounts) o;
        return resident == that.resident && fetching == that.fetching && skipped == that.skipped;
    }

    @Override
    public int hashCode() {
        return Long.hashCode(resident) * 31 * 31 + Long.hashCode(fetching) * 31 + Long.hashCode(skipped);
    }

    @Override
    public String toString() {
        return "TieredPrefetchOutcomeCounts{resident=" + resident + ", fetching=" + fetching + ", skipped=" + skipped + '}';
    }
}
