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
import org.elasticsearch.core.TieredPrefetchInput.Outcome;
import org.elasticsearch.test.ESTestCase;

import java.io.IOException;
import java.util.ArrayList;
import java.util.List;
import java.util.function.IntFunction;

import static org.hamcrest.Matchers.equalTo;
import static org.hamcrest.Matchers.greaterThanOrEqualTo;
import static org.hamcrest.Matchers.lessThanOrEqualTo;

public class PrefetchingCentroidIteratorTests extends ESTestCase {

    /**
     * A plain input must see exactly the historical behaviour: a fixed depth, one {@code prefetch} per posting list in
     * delegate order, issued when the posting list enters the buffer.
     */
    public void testFallbackUsesFixedDepthPrefetch() throws IOException {
        int numPostings = randomIntBetween(0, 50);
        int depth = randomIntBetween(1, 8);
        List<PostingMetadata> postings = postings(numPostings);
        RecordingIndexInput input = new RecordingIndexInput();

        // depth 1 also exercises the 2-arg constructor used by all production callers
        PrefetchingCentroidIterator iterator = depth == 1 && randomBoolean()
            ? new PrefetchingCentroidIterator(new ListCentroidIterator(postings), input)
            : new PrefetchingCentroidIterator(new ListCentroidIterator(postings), input, depth);
        assertThat(input.prefetched.size(), equalTo(Math.min(depth, numPostings)));

        List<PostingMetadata> consumed = new ArrayList<>();
        while (iterator.hasNext()) {
            consumed.add(iterator.nextPosting());
            assertThat(input.prefetched.size(), equalTo(Math.min(depth + consumed.size(), numPostings)));
            assertThat(iterator.window(), equalTo(depth));
        }
        assertThat(consumed, equalTo(postings));
        assertThat(input.prefetched, equalTo(offsets(postings)));
        assertThat(input.prefetchedLengths, equalTo(lengths(postings)));
        expectThrows(IllegalStateException.class, iterator::nextPosting);
    }

    /**
     * The fallback path must not grow even when a large maximum is requested, because a plain input gives no signal.
     */
    public void testFallbackIgnoresMaximum() throws IOException {
        List<PostingMetadata> postings = postings(40);
        RecordingIndexInput input = new RecordingIndexInput();
        PrefetchingCentroidIterator iterator = new PrefetchingCentroidIterator(new ListCentroidIterator(postings), input, 2, 16);
        assertThat(input.prefetched.size(), equalTo(2));
        List<PostingMetadata> consumed = new ArrayList<>();
        while (iterator.hasNext()) {
            consumed.add(iterator.nextPosting());
            assertThat(input.prefetched.size(), equalTo(Math.min(2 + consumed.size(), postings.size())));
        }
        assertThat(consumed, equalTo(postings));
        assertThat(iterator.window(), equalTo(2));
    }

    /**
     * Every {@code FETCHING} outcome doubles the window, so a cold remote cache reaches the cap while the buffer is
     * first filled, and the window never exceeds it.
     */
    public void testWindowGrowsOnFetchingUpToCap() throws IOException {
        int numPostings = 100;
        int cap = PrefetchingCentroidIterator.DEFAULT_MAX_PREFETCH_AHEAD;
        List<PostingMetadata> postings = postings(numPostings);
        TieredRecordingIndexInput input = new TieredRecordingIndexInput(i -> Outcome.FETCHING);

        PrefetchingCentroidIterator iterator = new PrefetchingCentroidIterator(new ListCentroidIterator(postings), input);
        assertThat(iterator.window(), equalTo(cap));
        assertThat(input.requested.size(), equalTo(cap));

        List<PostingMetadata> consumed = new ArrayList<>();
        while (iterator.hasNext()) {
            consumed.add(iterator.nextPosting());
            assertThat(iterator.window(), equalTo(cap));
            assertThat(input.requested.size(), equalTo(Math.min(cap + consumed.size(), numPostings)));
        }
        assertThat(consumed, equalTo(postings));
        assertThat(input.requested, equalTo(offsets(postings)));
        assertThat(input.prefetchCalls, equalTo(0));
    }

    /**
     * Growth follows the doubling sequence from the initial depth, and a streak of resident outcomes halves it again.
     */
    public void testWindowDoublesPerFetching() throws IOException {
        List<PostingMetadata> postings = postings(30);
        TieredRecordingIndexInput input = new TieredRecordingIndexInput(i -> i < 3 ? Outcome.FETCHING : Outcome.RESIDENT);
        PrefetchingCentroidIterator iterator = new PrefetchingCentroidIterator(new ListCentroidIterator(postings), input, 1, 64);
        // Requests 0..2 are FETCHING: 1 -> 2 -> 4 -> 8. Requests 3..6 are RESIDENT: the fourth halves 8 -> 4, at which
        // point the buffer (7 entries) is already above the window and filling stops.
        assertThat(input.requested.size(), equalTo(7));
        assertThat(iterator.window(), equalTo(4));
        assertThat(drain(iterator), equalTo(postings));
        assertThat(input.requested, equalTo(offsets(postings)));
    }

    /**
     * A streak of {@code RESIDENT} outcomes halves the window toward the initial depth but never below it, and the
     * iterator stops pulling from the delegate until the buffer drains below the smaller window.
     */
    public void testWindowShrinksOnResidentStreakNotBelowInitial() throws IOException {
        int initial = 2;
        int cap = 16;
        List<PostingMetadata> postings = postings(60);
        TieredRecordingIndexInput input = new TieredRecordingIndexInput(i -> i < 3 ? Outcome.FETCHING : Outcome.RESIDENT);
        PrefetchingCentroidIterator iterator = new PrefetchingCentroidIterator(new ListCentroidIterator(postings), input, initial, cap);

        // Requests 0..2 grow 2 -> 4 -> 8 -> 16; requests 3..6 are resident and halve it to 8; request 7 fills the
        // buffer to the new window.
        assertThat(input.requested.size(), equalTo(8));
        assertThat(iterator.window(), equalTo(8));

        List<PostingMetadata> consumed = new ArrayList<>();
        int previousWindow = iterator.window();
        while (iterator.hasNext()) {
            consumed.add(iterator.nextPosting());
            int window = iterator.window();
            assertThat("window only shrinks once all outcomes are resident", window, lessThanOrEqualTo(previousWindow));
            assertThat(window, greaterThanOrEqualTo(initial));
            assertThat("buffer never exceeds the largest window", input.requested.size() - consumed.size(), lessThanOrEqualTo(8));
            previousWindow = window;
        }
        assertThat(iterator.window(), equalTo(initial));
        assertThat(consumed, equalTo(postings));
        assertThat(input.requested, equalTo(offsets(postings)));
    }

    /**
     * At the floor, further resident streaks keep the window at the initial depth.
     */
    public void testAllResidentStaysAtInitialDepth() throws IOException {
        int initial = randomIntBetween(1, 4);
        List<PostingMetadata> postings = postings(50);
        TieredRecordingIndexInput input = new TieredRecordingIndexInput(i -> Outcome.RESIDENT);
        PrefetchingCentroidIterator iterator = new PrefetchingCentroidIterator(new ListCentroidIterator(postings), input, initial);
        assertThat(input.requested.size(), equalTo(initial));
        List<PostingMetadata> consumed = new ArrayList<>();
        while (iterator.hasNext()) {
            consumed.add(iterator.nextPosting());
            assertThat(iterator.window(), equalTo(initial));
            assertThat(input.requested.size(), equalTo(Math.min(initial + consumed.size(), postings.size())));
        }
        assertThat(consumed, equalTo(postings));
    }

    /**
     * {@code SKIPPED} means nothing more will be scheduled, so it must not deepen the window.
     */
    public void testNoGrowthOnSkipped() throws IOException {
        int initial = randomIntBetween(1, 4);
        List<PostingMetadata> postings = postings(40);
        TieredRecordingIndexInput input = new TieredRecordingIndexInput(i -> Outcome.SKIPPED);
        PrefetchingCentroidIterator iterator = new PrefetchingCentroidIterator(new ListCentroidIterator(postings), input, initial);
        assertThat(input.requested.size(), equalTo(initial));

        List<PostingMetadata> consumed = new ArrayList<>();
        while (iterator.hasNext()) {
            consumed.add(iterator.nextPosting());
            assertThat(iterator.window(), equalTo(initial));
            assertThat(input.requested.size(), equalTo(Math.min(initial + consumed.size(), postings.size())));
        }
        assertThat(consumed, equalTo(postings));
        assertThat(input.requested, equalTo(offsets(postings)));
        assertThat(input.prefetchCalls, equalTo(0));
    }

    /**
     * {@code SKIPPED} keeps a grown window as it is and breaks a {@code RESIDENT} streak.
     */
    public void testSkippedKeepsWindowAndResetsResidentStreak() throws IOException {
        // FETCHING grows 1 -> 2; then R, R, R, S, R, R, R never reaches a streak of four.
        Outcome[] script = new Outcome[] {
            Outcome.FETCHING,
            Outcome.RESIDENT,
            Outcome.RESIDENT,
            Outcome.RESIDENT,
            Outcome.SKIPPED,
            Outcome.RESIDENT,
            Outcome.RESIDENT,
            Outcome.RESIDENT };
        List<PostingMetadata> postings = postings(script.length);
        TieredRecordingIndexInput input = new TieredRecordingIndexInput(i -> script[i]);
        PrefetchingCentroidIterator iterator = new PrefetchingCentroidIterator(new ListCentroidIterator(postings), input);
        assertThat(iterator.window(), equalTo(2));
        List<PostingMetadata> consumed = new ArrayList<>();
        while (iterator.hasNext()) {
            consumed.add(iterator.nextPosting());
            assertThat(iterator.window(), equalTo(2));
        }
        assertThat(consumed, equalTo(postings));
        assertThat(input.requested, equalTo(offsets(postings)));
    }

    /**
     * Whatever the outcomes, the iterator returns exactly the delegate's posting lists in order, requests each one
     * exactly once via {@code ensureResident} (never {@code prefetch}), and keeps the window and buffer within bounds.
     */
    public void testRandomOutcomesPreserveOrderAndRequestEachOnce() throws IOException {
        int numPostings = randomIntBetween(0, 200);
        int initial = randomIntBetween(1, 4);
        int cap = randomIntBetween(initial, 32);
        List<PostingMetadata> postings = postings(numPostings);
        Outcome[] script = new Outcome[numPostings];
        for (int i = 0; i < numPostings; i++) {
            script[i] = randomFrom(Outcome.values());
        }
        TieredRecordingIndexInput input = new TieredRecordingIndexInput(i -> script[i]);
        PrefetchingCentroidIterator iterator = new PrefetchingCentroidIterator(new ListCentroidIterator(postings), input, initial, cap);

        List<PostingMetadata> consumed = new ArrayList<>();
        assertBounds(iterator, input, consumed, initial, cap);
        while (iterator.hasNext()) {
            consumed.add(iterator.nextPosting());
            assertBounds(iterator, input, consumed, initial, cap);
        }
        assertThat(consumed, equalTo(postings));
        assertThat(input.requested, equalTo(offsets(postings)));
        assertThat(input.requestedLengths, equalTo(lengths(postings)));
        assertThat(input.prefetchCalls, equalTo(0));
        expectThrows(IllegalStateException.class, iterator::nextPosting);
    }

    public void testThreeArgConstructorWithDepthAboveDefaultCap() throws IOException {
        int depth = PrefetchingCentroidIterator.DEFAULT_MAX_PREFETCH_AHEAD + randomIntBetween(1, 8);
        List<PostingMetadata> postings = postings(100);
        TieredRecordingIndexInput input = new TieredRecordingIndexInput(i -> Outcome.FETCHING);
        PrefetchingCentroidIterator iterator = new PrefetchingCentroidIterator(new ListCentroidIterator(postings), input, depth);
        assertThat(iterator.window(), equalTo(depth));
        assertThat(input.requested.size(), equalTo(depth));
        assertThat(drain(iterator), equalTo(postings));
    }

    public void testInvalidArguments() {
        CentroidIterator empty = new ListCentroidIterator(List.of());
        RecordingIndexInput input = new RecordingIndexInput();
        expectThrows(IllegalArgumentException.class, () -> new PrefetchingCentroidIterator(empty, input, 0));
        expectThrows(IllegalArgumentException.class, () -> new PrefetchingCentroidIterator(empty, input, 4, 3));
    }

    private static void assertBounds(
        PrefetchingCentroidIterator iterator,
        TieredRecordingIndexInput input,
        List<PostingMetadata> consumed,
        int initial,
        int cap
    ) {
        assertThat(iterator.window(), greaterThanOrEqualTo(initial));
        assertThat(iterator.window(), lessThanOrEqualTo(cap));
        int buffered = input.requested.size() - consumed.size();
        assertThat(buffered, lessThanOrEqualTo(cap));
        assertThat(buffered, greaterThanOrEqualTo(0));
    }

    private static List<PostingMetadata> drain(PrefetchingCentroidIterator iterator) throws IOException {
        List<PostingMetadata> consumed = new ArrayList<>();
        while (iterator.hasNext()) {
            consumed.add(iterator.nextPosting());
        }
        return consumed;
    }

    private static List<PostingMetadata> postings(int count) {
        List<PostingMetadata> postings = new ArrayList<>(count);
        long offset = randomLongBetween(0, 1000);
        for (int i = 0; i < count; i++) {
            long length = randomLongBetween(1, 4096);
            postings.add(new PostingMetadata(offset, length, i, randomFloat()));
            offset += length;
        }
        return postings;
    }

    private static List<Long> offsets(List<PostingMetadata> postings) {
        return postings.stream().map(PostingMetadata::offset).toList();
    }

    private static List<Long> lengths(List<PostingMetadata> postings) {
        return postings.stream().map(PostingMetadata::length).toList();
    }

    /** Plain delegate over a fixed list, standing in for the score-ordered centroid iterators of the readers. */
    private static final class ListCentroidIterator implements CentroidIterator {
        private final List<PostingMetadata> postings;
        private int next = 0;

        ListCentroidIterator(List<PostingMetadata> postings) {
            this.postings = postings;
        }

        @Override
        public boolean hasNext() {
            return next < postings.size();
        }

        @Override
        public PostingMetadata nextPosting() {
            return postings.get(next++);
        }
    }

    /**
     * An {@link IndexInput} that records {@code prefetch} calls and holds no data. {@link PrefetchingCentroidIterator}
     * never reads from its input, so only the hint calls matter, and a real directory input would not let the test
     * observe them.
     */
    private static class RecordingIndexInput extends IndexInput {
        final List<Long> prefetched = new ArrayList<>();
        final List<Long> prefetchedLengths = new ArrayList<>();
        int prefetchCalls = 0;

        RecordingIndexInput() {
            super("recording");
        }

        @Override
        public void prefetch(long offset, long length) {
            prefetchCalls++;
            prefetched.add(offset);
            prefetchedLengths.add(length);
        }

        @Override
        public void close() {}

        @Override
        public long getFilePointer() {
            return 0;
        }

        @Override
        public void seek(long pos) {
            throw new UnsupportedOperationException();
        }

        @Override
        public long length() {
            return Long.MAX_VALUE;
        }

        @Override
        public IndexInput slice(String sliceDescription, long offset, long length) {
            throw new UnsupportedOperationException();
        }

        @Override
        public byte readByte() {
            throw new UnsupportedOperationException();
        }

        @Override
        public void readBytes(byte[] b, int offset, int len) {
            throw new UnsupportedOperationException();
        }
    }

    /**
     * A {@link TieredPrefetchInput} double with scripted outcomes. It is necessary because the real implementation
     * lives in the stateless plugin, which server tests cannot depend on, and because the adaptive policy can only be
     * tested deterministically if the outcome of each request is controlled. The script is indexed by the order of
     * the {@code ensureResident} call, which equals the delegate order of the posting list being requested.
     */
    private static final class TieredRecordingIndexInput extends RecordingIndexInput implements TieredPrefetchInput {
        private final IntFunction<Outcome> script;
        final List<Long> requested = new ArrayList<>();
        final List<Long> requestedLengths = new ArrayList<>();

        TieredRecordingIndexInput(IntFunction<Outcome> script) {
            this.script = script;
        }

        @Override
        public Outcome ensureResident(long offset, long length) {
            Outcome outcome = script.apply(requested.size());
            requested.add(offset);
            requestedLengths.add(length);
            return outcome;
        }

        @Override
        public void ensureResident(long[] offsets, int length, int count, Outcome[] outcomes) {
            throw new AssertionError("PrefetchingCentroidIterator requests posting lists one at a time");
        }

        @Override
        public long residencyRegionSize() {
            return 16 * 1024 * 1024;
        }
    }
}
