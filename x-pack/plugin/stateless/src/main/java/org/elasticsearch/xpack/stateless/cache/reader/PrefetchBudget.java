/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the Elastic License
 * 2.0; you may not use this file except in compliance with the Elastic License
 * 2.0.
 */

package org.elasticsearch.xpack.stateless.cache.reader;

import java.util.HashSet;
import java.util.Set;

/**
 * Node-scoped bound on how many cache regions {@link CacheFileReader#ensureResident} may have in flight from the object store
 * at once.
 *
 * <p>{@code ensureResident} exists so callers can prefetch far more aggressively than {@code IndexInput#prefetch}. Without a
 * bound, a single kNN rescore over scattered vectors could put hundreds of region fetches in flight, saturating the shard-read
 * pool that demand reads also depend on. {@link FillCacheMemoryPressure} cannot serve as that bound: it queues acquirers
 * rather than rejecting them, and a prefetch must never block. This budget is the non-blocking counterpart: a caller either
 * gets admitted immediately or is told {@link org.elasticsearch.core.TieredPrefetchInput.Outcome#SKIPPED}, which is itself the
 * signal to stop deepening its prefetch window.
 *
 * <p>The unit is regions rather than bytes because every admitted fetch fills exactly one region, and the region size is fixed
 * per node, so a region count is a byte bound with fewer moving parts.
 *
 * <p>Regions are tracked by identity so that a second request for a region whose fetch this budget already admitted
 * {@link Admission#JOINED joins} it without consuming a slot. A region being filled by a demand read rather than by a prefetch
 * is not visible here and does consume a slot until the shared fill completes; that slot is released by the same listener, so
 * the over-count is short-lived.
 *
 * <p>Plain {@code IndexInput#prefetch} is deliberately not subject to this budget so its behaviour is unchanged.
 */
public final class PrefetchBudget {

    /** A budget that never denies. Used where no bound is configured, such as tests and sub-file copies. */
    public static final PrefetchBudget UNLIMITED = new PrefetchBudget(Integer.MAX_VALUE);

    /** Result of {@link #tryAcquire}. */
    public enum Admission {
        /** A slot was taken for this region. The caller owns it and must {@link #release} it when the fetch settles. */
        ACQUIRED,
        /** This region is already in flight under this budget. No slot was taken and nothing needs releasing. */
        JOINED,
        /** No slot is available. Nothing was taken. */
        DENIED
    }

    /** Identifies one region of one cache file. {@code cacheKey} is whatever the cache file reports as its key. */
    public record RegionKey(Object cacheKey, int region) {}

    private final int maxInFlightRegions;
    private final Set<RegionKey> inFlight = new HashSet<>();

    public PrefetchBudget(int maxInFlightRegions) {
        if (maxInFlightRegions < 0) {
            throw new IllegalArgumentException("maxInFlightRegions must not be negative, got [" + maxInFlightRegions + "]");
        }
        this.maxInFlightRegions = maxInFlightRegions;
    }

    /**
     * Attempts to admit a fetch for {@code key}. Never blocks. Synchronized rather than lock-free because the decision must be
     * atomic across "already present", "under limit" and "insert", and it runs once per region per call, not per byte.
     */
    public synchronized Admission tryAcquire(RegionKey key) {
        if (inFlight.contains(key)) {
            return Admission.JOINED;
        }
        if (inFlight.size() >= maxInFlightRegions) {
            return Admission.DENIED;
        }
        inFlight.add(key);
        return Admission.ACQUIRED;
    }

    /**
     * Returns the slot taken by an earlier {@link Admission#ACQUIRED} for {@code key}. Must be called exactly once per
     * acquisition, whether the fetch succeeded or failed.
     */
    public synchronized void release(RegionKey key) {
        final boolean removed = inFlight.remove(key);
        assert removed : "released a region that was not in flight: " + key;
    }

    /** Number of regions currently admitted. */
    public synchronized int inFlightRegions() {
        return inFlight.size();
    }

    public int maxInFlightRegions() {
        return maxInFlightRegions;
    }

    @Override
    public String toString() {
        return "PrefetchBudget{inFlight=" + inFlightRegions() + ", max=" + maxInFlightRegions + '}';
    }
}
