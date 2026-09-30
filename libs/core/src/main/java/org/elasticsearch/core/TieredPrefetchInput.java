/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the "Elastic License
 * 2.0", the "GNU Affero General Public License v3.0 only", and the "Server Side
 * Public License v 1"; you may not use this file except in compliance with, at
 * your election, the "Elastic License 2.0", the "GNU Affero General Public
 * License v3.0 only", or the "Server Side Public License, v 1".
 */

package org.elasticsearch.core;

import java.io.IOException;

/**
 * An optional interface that an {@code IndexInput} can implement when its bytes live in a tiered store: a local cache
 * (disk and page cache) in front of a remote tier such as an object store, where a remote round trip costs orders of
 * magnitude more than a local read.
 *
 * <p>Lucene's {@code IndexInput#prefetch} is a fire-and-forget hint. Implementations backed by a tiered store already
 * use it to start remote fetches, but the caller learns nothing about whether the bytes were local, so it cannot
 * adapt: it prefetches from the remote tier exactly as far ahead as it would from the page cache. This interface adds
 * the missing feedback. Callers that drive a prefetch-then-read loop (posting list iteration, vector rescoring) can
 * deepen their window when bytes are remote, keep it shallow when bytes are local, and stop when the implementation
 * reports it will not schedule more work.
 *
 * <p>Every method here is non-blocking and every {@link Outcome} is a snapshot. A range reported as
 * {@link Outcome#FETCHING} may be resident by the time it is read, and a {@link Outcome#RESIDENT} range may have
 * been evicted. Correctness never depends on the outcome: a subsequent read always falls back to a blocking fetch.
 * Callers must treat outcomes purely as scheduling hints.
 *
 * <p>Wrappers that delegate to another input (metrics, reopening, filtering) must forward these methods explicitly.
 * Lucene's {@code FilterIndexInput} does not forward {@code prefetch}, and it will not forward this either.
 *
 * <p>Consuming bytes as they arrive from the remote tier, rather than waiting for them to be written to the local
 * cache and then reading them back, is a good alternative to this design. It is deliberately not attempted here:
 * the existing callers are all structured as hint-then-blocking-read, and moving them to an asynchronous model is a
 * much deeper change than adding feedback to the hint.
 *
 * <p>Upstream Lucene has changed {@code IndexInput#prefetch} to return {@code boolean}, true when the call
 * actually scheduled a fetch (GITHUB#15627). Once Elasticsearch is on that version, an implementation of this
 * interface can satisfy that contract with {@code ensureResident(offset, length) != Outcome.RESIDENT}.
 */
public interface TieredPrefetchInput {

    /**
     * What happened to a request to make a range resident. Ordered from cheapest to most expensive for the caller
     * to read next.
     */
    enum Outcome {
        /**
         * The bytes are already in the local tier. Reading them now is cheap. No work was scheduled.
         */
        RESIDENT,
        /**
         * The bytes were not local. A remote fetch was started, or an in-flight fetch for the same region was joined.
         * Reading them now blocks for a remote round trip. Reading them later may not.
         */
        FETCHING,
        /**
         * The bytes were not local and nothing was scheduled: the implementation's fetch budget is exhausted, the
         * cache could not make room, or the range is outside what this input can prefetch. Reading them will block
         * for a remote round trip whenever it happens. Callers should not deepen their prefetch window in response.
         */
        SKIPPED
    }

    /**
     * Asks the input to make {@code [offset, offset + length)} resident in the local tier, without blocking.
     *
     * <p>Implementations coalesce the request to their own cache granularity (see {@link #residencyRegionSize()}), so
     * asking for a few bytes may fetch far more. When the range spans several regions in different states, the
     * returned outcome is the most expensive of them: any region fetching makes the whole range {@link Outcome#FETCHING},
     * and any region skipped makes it {@link Outcome#SKIPPED}.
     *
     * <p>Like {@code IndexInput#prefetch}, this tolerates ranges that do not lie within the input: a range starting
     * outside it, or with a non-positive length, is reported as {@link Outcome#SKIPPED} rather than rejected, since the
     * outcome is a hint and must never fail a read.
     *
     * @param offset the byte offset within this input
     * @param length the number of bytes requested
     * @return the snapshot outcome for the range
     */
    Outcome ensureResident(long offset, long length) throws IOException;

    /**
     * Bulk form of {@link #ensureResident(long, long)} for scattered fixed-size records such as vectors addressed by
     * ordinal. Writes the outcome for each {@code offsets[i]} into {@code outcomes[i]} for {@code i < count}.
     *
     * <p>Implementations group the ranges by region before touching the cache, so all ranges within one region share
     * one outcome and cost one lookup, regardless of how many ranges fall into it. The per-range result lets a caller
     * map outcomes back to its own records without knowing the region layout. Ranges that themselves span a region
     * boundary follow the rule in {@link #ensureResident(long, long)}, and ranges starting outside the input are reported
     * {@link Outcome#SKIPPED}. A non-positive {@code length} is an argument error, since it applies to every range.
     *
     * @param offsets  byte offsets within this input for each range, as in {@link #ensureResident(long, long)};
     *                 only {@code [0, count)} are read
     * @param length   byte length of each range (same for all); must be positive
     * @param count    number of ranges
     * @param outcomes output array; must have at least {@code count} entries, and only {@code [0, count)} are written
     */
    void ensureResident(long[] offsets, int length, int count, Outcome[] outcomes) throws IOException;

    /**
     * The granularity, in bytes, at which this input's local tier tracks residency and fetches from the remote tier.
     * Two offsets with the same {@code offset / residencyRegionSize()} live in the same region and will always share
     * an outcome. Callers can use this to group their own records before calling {@link #ensureResident(long, long)}
     * one region at a time, or to reason about how much a small request will actually fetch.
     *
     * @return the region size in bytes; always positive
     */
    long residencyRegionSize();

    /**
     * Validates the arguments to {@link #ensureResident(long[], int, int, Outcome[])}. Returns {@code true} when
     * {@code count} is zero and the caller should treat the call as a no-op.
     */
    static boolean checkBulkArgs(long[] offsets, int length, int count, Outcome[] outcomes) {
        if (count < 0) {
            throw new IllegalArgumentException("count must not be negative, got [" + count + "]");
        }
        if (length <= 0) {
            throw new IllegalArgumentException("length must be positive, got [" + length + "]");
        }
        if (offsets.length < count) {
            throw new IllegalArgumentException("offsets array length [" + offsets.length + "] is less than count [" + count + "]");
        }
        if (outcomes.length < count) {
            throw new IllegalArgumentException("outcomes array length [" + outcomes.length + "] is less than count [" + count + "]");
        }
        return count == 0;
    }
}
