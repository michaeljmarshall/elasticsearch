/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the "Elastic License
 * 2.0", the "GNU Affero General Public License v3.0 only", and the "Server Side
 * Public License v 1"; you may not use this file except in compliance with, at
 * your election, the "Elastic License 2.0", the "GNU Affero General Public
 * License v3.0 only", or the "Server Side Public License, v 1".
 */

package org.elasticsearch.index.codec.vectors.diskbbq;

import org.apache.lucene.store.IndexInput;
import org.elasticsearch.core.TieredPrefetchInput;

import java.io.IOException;

/**
 * A configurable iterator that prefetches posting lists ahead of consumption
 * to optimize disk I/O performance. This iterator wraps another CentroidIterator
 * and maintains a buffer of prefetched posting list locations.
 *
 * <p>When the posting list input is a plain {@link IndexInput}, the buffer has a fixed depth and each posting list is
 * hinted with {@link IndexInput#prefetch}. That depth is tuned for local storage, where a miss costs a page-cache or
 * disk read.
 *
 * <p>When the input implements {@link TieredPrefetchInput}, its bytes may live in a remote tier where a miss costs a
 * round trip that is orders of magnitude more expensive, so a fixed shallow depth leaves the caller waiting on one
 * remote fetch at a time. In that mode each posting list is requested with
 * {@link TieredPrefetchInput#ensureResident(long, long)} instead, and the depth adapts to the outcomes:
 * <ul>
 *     <li>{@link TieredPrefetchInput.Outcome#FETCHING}: the window doubles, up to the maximum depth, so that more
 *     remote fetches overlap with consumption. It doubles at most once per consumed posting list, so a cold cache
 *     widens the window as the search progresses rather than claiming the whole prefetch budget up front.</li>
 *     <li>{@link TieredPrefetchInput.Outcome#RESIDENT}: after {@link #RESIDENT_STREAK_TO_SHRINK} consecutive resident
 *     outcomes the window halves, never below the initial depth, so a warm cache does not pull centroids from the
 *     delegate far beyond what the caller will consume.</li>
 *     <li>{@link TieredPrefetchInput.Outcome#SKIPPED}: the window stays as it is. The input will not schedule more
 *     work, so looking further ahead would not start any more fetches.</li>
 * </ul>
 * Outcomes are only scheduling hints; the order and set of posting lists returned is identical in both modes.
 *
 * The iterator is not thread-safe and is designed for single-threaded access.
 */
public final class PrefetchingCentroidIterator implements CentroidIterator {

    /**
     * Default upper bound on the adaptive window when the input is a {@link TieredPrefetchInput}. It bounds the ring
     * allocation per iterator and how many posting lists beyond the caller's visit budget may be fetched in vain.
     */
    public static final int DEFAULT_MAX_PREFETCH_AHEAD = 16;

    /**
     * Number of consecutive {@link TieredPrefetchInput.Outcome#RESIDENT} outcomes after which the adaptive window is
     * halved. Small enough to back off quickly once the cache is warm, large enough that a single resident posting
     * list among remote ones does not collapse the window.
     */
    static final int RESIDENT_STREAK_TO_SHRINK = 4;

    private final CentroidIterator delegate;
    private final IndexInput postingListSlice;
    /** Non-null when {@link #postingListSlice} reports residency outcomes; selects the adaptive path. */
    private final TieredPrefetchInput tieredInput;
    private final int initialPrefetchAhead;
    private final int maxPrefetchAhead;

    // Current target number of buffered posting lists; always within [initialPrefetchAhead, maxPrefetchAhead]
    private int window;
    // Consecutive RESIDENT outcomes since the last window change or non-resident outcome
    private int residentStreak = 0;
    // Whether the window has already doubled since the caller last consumed a posting list
    private boolean grownSinceLastConsume = false;

    // Ring buffer for prefetched offsets and lengths
    private final PostingMetadata[] prefetchBuffer;
    private int readIndex = 0;  // Where we read from buffer
    private int writeIndex = 0; // Where we write to buffer
    private int bufferCount = 0; // Number of elements in buffer

    /**
     * Creates a prefetching iterator with default prefetch depth of 1.
     *
     * @param delegate the underlying centroid iterator
     * @param postingListSlice the index input for posting lists
     * @throws IOException if prefetching fails during initialization
     */
    public PrefetchingCentroidIterator(CentroidIterator delegate, IndexInput postingListSlice) throws IOException {
        this(delegate, postingListSlice, 1);
    }

    /**
     * Creates a prefetching iterator with configurable prefetch depth. On a {@link TieredPrefetchInput} the depth is
     * the initial and minimum depth, and the window may grow to {@link #DEFAULT_MAX_PREFETCH_AHEAD} (or
     * {@code prefetchAhead} if larger). Otherwise it is the fixed depth.
     *
     * @param delegate the underlying centroid iterator
     * @param postingListSlice the index input for posting lists
     * @param prefetchAhead number of posting lists to prefetch ahead (must be &gt;= 1)
     * @throws IOException if prefetching fails during initialization
     * @throws IllegalArgumentException if {@code prefetchAhead < 1}
     */
    public PrefetchingCentroidIterator(CentroidIterator delegate, IndexInput postingListSlice, int prefetchAhead) throws IOException {
        this(delegate, postingListSlice, prefetchAhead, Math.max(prefetchAhead, DEFAULT_MAX_PREFETCH_AHEAD));
    }

    /**
     * Creates a prefetching iterator with an explicit bound on the adaptive window.
     *
     * @param delegate the underlying centroid iterator
     * @param postingListSlice the index input for posting lists
     * @param prefetchAhead initial and minimum depth on a {@link TieredPrefetchInput}, fixed depth otherwise
     *                      (must be &gt;= 1)
     * @param maxPrefetchAhead maximum depth on a {@link TieredPrefetchInput}; ignored otherwise
     *                         (must be &gt;= {@code prefetchAhead})
     * @throws IOException if prefetching fails during initialization
     * @throws IllegalArgumentException if {@code prefetchAhead < 1} or {@code maxPrefetchAhead < prefetchAhead}
     */
    public PrefetchingCentroidIterator(CentroidIterator delegate, IndexInput postingListSlice, int prefetchAhead, int maxPrefetchAhead)
        throws IOException {
        if (prefetchAhead < 1) {
            throw new IllegalArgumentException("prefetchAhead must be at least 1, got: " + prefetchAhead);
        }
        if (maxPrefetchAhead < prefetchAhead) {
            throw new IllegalArgumentException(
                "maxPrefetchAhead must be at least prefetchAhead [" + prefetchAhead + "], got: " + maxPrefetchAhead
            );
        }
        this.delegate = delegate;
        this.postingListSlice = postingListSlice;
        this.tieredInput = postingListSlice instanceof TieredPrefetchInput tiered ? tiered : null;
        this.initialPrefetchAhead = prefetchAhead;
        // The fallback path never grows, so it only needs a ring of the fixed depth.
        this.maxPrefetchAhead = tieredInput != null ? maxPrefetchAhead : prefetchAhead;
        this.window = prefetchAhead;
        this.prefetchBuffer = new PostingMetadata[this.maxPrefetchAhead];
        // Initialize buffer by prefetching up to the initial window
        fillBuffer();
    }

    /**
     * Fills the prefetch buffer up to the current window. The window may grow while filling, in which case filling
     * continues up to the new window.
     */
    private void fillBuffer() throws IOException {
        while (bufferCount < window && delegate.hasNext()) {
            PostingMetadata offsetAndLength = delegate.nextPosting();
            prefetchBuffer[writeIndex] = offsetAndLength;
            writeIndex = (writeIndex + 1) % prefetchBuffer.length;
            bufferCount++;

            requestPostingList(offsetAndLength);
        }
    }

    /**
     * Hints the input about a posting list that just entered the buffer, adapting the window on the tiered path.
     */
    private void requestPostingList(PostingMetadata postingMetadata) throws IOException {
        if (tieredInput == null) {
            postingListSlice.prefetch(postingMetadata.offset(), postingMetadata.length());
            return;
        }
        TieredPrefetchInput.Outcome outcome = tieredInput.ensureResident(postingMetadata.offset(), postingMetadata.length());
        switch (outcome) {
            case FETCHING -> {
                residentStreak = 0;
                // Grow at most once per consumed posting list. Growing on every FETCHING outcome while filling would let a
                // cold cache jump straight to the cap before anything is consumed, taking that many budget slots from the
                // node-wide prefetch budget that the rescore phase of the same search also needs.
                if (grownSinceLastConsume == false) {
                    grownSinceLastConsume = true;
                    window = Math.min(window * 2, maxPrefetchAhead);
                }
            }
            case RESIDENT -> {
                if (++residentStreak >= RESIDENT_STREAK_TO_SHRINK) {
                    residentStreak = 0;
                    window = Math.max(window / 2, initialPrefetchAhead);
                }
            }
            case SKIPPED -> residentStreak = 0;
        }
    }

    /** The current target number of buffered posting lists. Visible for testing. */
    int window() {
        return window;
    }

    @Override
    public boolean hasNext() {
        return bufferCount > 0;
    }

    @Override
    public PostingMetadata nextPosting() throws IOException {
        if (bufferCount == 0) {
            throw new IllegalStateException("No more elements available");
        }

        // Get the next element from buffer
        PostingMetadata result = prefetchBuffer[readIndex];
        prefetchBuffer[readIndex] = null;
        readIndex = (readIndex + 1) % prefetchBuffer.length;
        bufferCount--;

        // Refill the buffer to the current window; the window may double once more during this refill
        grownSinceLastConsume = false;
        fillBuffer();

        return result;
    }
}
