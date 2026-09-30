/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the "Elastic License
 * 2.0", the "GNU Affero General Public License v3.0 only", and the "Server Side
 * Public License v 1"; you may not use this file except in compliance with, at
 * your election, the "Elastic License 2.0", the "GNU Affero General Public
 * License v3.0 only", or the "Server Side Public License, v 1".
 */

package org.elasticsearch.search.profile.query;

import org.elasticsearch.core.TieredPrefetchInput;

/**
 * Tallies the {@link TieredPrefetchInput.Outcome}s of prefetch requests made by a query. The split between resident,
 * fetching and skipped ranges shows how much of a query's data had to come from the remote tier of a tiered store,
 * which is what decides whether deeper or earlier prefetching is worth it.
 *
 * <p>Not thread-safe: a query accumulates into its own instance on the thread that rewrites it.
 */
public final class TieredPrefetchOutcomeCounts {

    private long resident;
    private long fetching;
    private long skipped;

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

    @Override
    public String toString() {
        return "TieredPrefetchOutcomeCounts{resident=" + resident + ", fetching=" + fetching + ", skipped=" + skipped + '}';
    }
}
