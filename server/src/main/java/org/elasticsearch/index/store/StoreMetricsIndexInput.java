/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the "Elastic License
 * 2.0", the "GNU Affero General Public License v3.0 only", and the "Server Side
 * Public License v 1"; you may not use this file except in compliance with, at
 * your election, the "Elastic License 2.0", the "GNU Affero General Public
 * License v3.0 only", or the "Server Side Public License, v 1".
 */

package org.elasticsearch.index.store;

import org.apache.lucene.store.FilterIndexInput;
import org.apache.lucene.store.IOContext;
import org.apache.lucene.store.IndexInput;
import org.apache.lucene.store.MemorySegmentAccessInput;
import org.apache.lucene.store.RandomAccessInput;
import org.elasticsearch.core.CheckedConsumer;
import org.elasticsearch.core.DirectAccessInput;
import org.elasticsearch.core.TieredPrefetchInput;
import org.elasticsearch.lucene.store.MemorySegmentAccessInputAccess;

import java.io.IOException;
import java.lang.foreign.MemorySegment;
import java.util.Map;
import java.util.Optional;
import java.util.Set;

public class StoreMetricsIndexInput extends FilterIndexInput implements DirectAccessInput {
    final PluggableDirectoryMetricsHolder<StoreMetrics> metricHolder;

    public static IndexInput create(String resourceDescription, IndexInput in, PluggableDirectoryMetricsHolder<StoreMetrics> metricHolder) {
        if (in instanceof StoreMetricsIndexInput) {
            // annoyingly, source-only snapshots do this for linked files.
            return in;
        } else if (in instanceof SelfAccountingIndexInput selfAccounting) {
            selfAccounting.accountBytesReadTo(metricHolder);
            return in;
        } else {
            return wrap(resourceDescription, in, metricHolder);
        }
    }

    /**
     * Chooses the wrapper by the delegate's capabilities. {@link DirectAccessInput} is claimed unconditionally because its
     * methods return {@code false} when the delegate lacks it, which callers treat as "use the plain path". The
     * {@link TieredPrefetchInput} capability is different: callers stop issuing plain {@code prefetch} hints once they see a
     * tiered input, and every shard directory on local storage is wrapped for metrics, so a wrapper that always claimed it
     * would silently disable prefetch on local storage. It is therefore only claimed when the delegate has it. Slices and
     * clones re-dispatch through here, since a slice of a tiered input may come back as a plain heap buffer.
     */
    private static StoreMetricsIndexInput wrap(
        String resourceDescription,
        IndexInput in,
        PluggableDirectoryMetricsHolder<StoreMetrics> metricHolder
    ) {
        if (in instanceof RandomAccessInput) {
            return in instanceof TieredPrefetchInput
                ? new TieredRandomAccessIndexInput(resourceDescription, in, metricHolder)
                : new RandomAccessIndexInput(resourceDescription, in, metricHolder);
        }
        return in instanceof TieredPrefetchInput
            ? new TieredStoreMetricsIndexInput(resourceDescription, in, metricHolder)
            : new StoreMetricsIndexInput(resourceDescription, in, metricHolder);
    }

    private StoreMetricsIndexInput(String resourceDescription, IndexInput in, PluggableDirectoryMetricsHolder<StoreMetrics> metricHolder) {
        super(resourceDescription, in);
        this.metricHolder = metricHolder;
        assert in instanceof StoreMetricsIndexInput == false;
    }

    @Override
    public byte readByte() throws IOException {
        byte result = in.readByte();
        addBytesRead(1);
        return result;
    }

    @Override
    public void readBytes(byte[] b, int offset, int len) throws IOException {
        in.readBytes(b, offset, len);
        addBytesRead(len);
    }

    final IndexInput createCopy(String resourceDescription, IndexInput in, PluggableDirectoryMetricsHolder<StoreMetrics> metricHolder) {
        return wrap(resourceDescription, in, metricHolder);
    }

    @Override
    public IndexInput clone() {
        return createCopy(toString(), in.clone(), metricHolder.singleThreaded());
    }

    @Override
    public IndexInput slice(String sliceDescription, long offset, long length) throws IOException {
        return createCopy(sliceDescription, in.slice(sliceDescription, offset, length), metricHolder.singleThreaded());
    }

    @Override
    public IndexInput slice(String sliceDescription, long offset, long length, IOContext context) throws IOException {
        return createCopy(sliceDescription, in.slice(sliceDescription, offset, length, context), metricHolder.singleThreaded());
    }

    @Override
    public RandomAccessInput randomAccessSlice(long offset, long length) throws IOException {
        RandomAccessInput delegate = in.randomAccessSlice(offset, length);
        if (delegate instanceof IndexInput input) {
            return (RandomAccessInput) wrap(input.toString(), input, metricHolder.singleThreaded());
        } else {
            return new MetricsRandomAccessInput(delegate, metricHolder.singleThreaded());
        }
    }

    @Override
    public void prefetch(long offset, long length) throws IOException {
        in.prefetch(offset, length);
    }

    @Override
    public boolean withMemorySegmentSlice(long offset, long length, CheckedConsumer<MemorySegment, IOException> action) throws IOException {
        if (in instanceof DirectAccessInput dai) {
            return dai.withMemorySegmentSlice(offset, length, action);
        }
        return false;
    }

    @Override
    public boolean withSliceAddresses(
        long[] offsets,
        int length,
        int count,
        MemorySegment addressesScratch,
        CheckedConsumer<MemorySegment, IOException> action
    ) throws IOException {
        if (in instanceof DirectAccessInput dai) {
            return dai.withSliceAddresses(offsets, length, count, addressesScratch, action);
        }
        return false;
    }

    @Override
    public Optional<Boolean> isLoaded() {
        return in.isLoaded();
    }

    @Override
    public void updateIOContext(IOContext context) throws IOException {
        in.updateIOContext(context);
    }

    void addBytesRead(long bytes) {
        metricHolder.instance().addBytesRead(bytes);
    }

    @Override
    public void readBytes(byte[] b, int offset, int len, boolean useBuffer) throws IOException {
        in.readBytes(b, offset, len, useBuffer);
        addBytesRead(len);
    }

    @Override
    public short readShort() throws IOException {
        short result = in.readShort();
        addBytesRead(2);
        return result;
    }

    @Override
    public int readInt() throws IOException {
        int result = getDelegate().readInt();
        addBytesRead(4);
        return result;
    }

    @Override
    public int readVInt() throws IOException {
        long position = in.getFilePointer();
        int result = in.readVInt();
        long bytes = in.getFilePointer() - position;
        assert bytes > 0;
        addBytesRead(bytes);
        return result;
    }

    @Override
    public int readZInt() throws IOException {
        long position = in.getFilePointer();
        int result = in.readZInt();
        long bytes = in.getFilePointer() - position;
        assert bytes > 0;
        addBytesRead(bytes);
        return result;
    }

    @Override
    public long readLong() throws IOException {
        long result = getDelegate().readLong();
        addBytesRead(8);
        return result;
    }

    @Override
    public void readLongs(long[] dst, int offset, int length) throws IOException {
        getDelegate().readLongs(dst, offset, length);
        metricHolder.instance().addBytesRead(8L * length);
    }

    @Override
    public void readInts(int[] dst, int offset, int length) throws IOException {
        getDelegate().readInts(dst, offset, length);
        metricHolder.instance().addBytesRead(4L * length);
    }

    @Override
    public void readFloats(float[] floats, int offset, int len) throws IOException {
        getDelegate().readFloats(floats, offset, len);
        metricHolder.instance().addBytesRead(4L * len);
    }

    @Override
    public long readVLong() throws IOException {
        long position = in.getFilePointer();
        long result = in.readVLong();
        long bytes = in.getFilePointer() - position;
        assert bytes > 0;
        addBytesRead(bytes);
        return result;
    }

    @Override
    public long readZLong() throws IOException {
        long position = in.getFilePointer();
        long result = in.readZLong();
        long bytes = in.getFilePointer() - position;
        assert bytes > 0;
        addBytesRead(bytes);

        return result;
    }

    @Override
    public String readString() throws IOException {
        long position = in.getFilePointer();
        String result = in.readString();
        long bytes = in.getFilePointer() - position;
        assert bytes > 0;
        addBytesRead(bytes);
        return result;
    }

    @Override
    public Map<String, String> readMapOfStrings() throws IOException {
        long position = in.getFilePointer();
        Map<String, String> result = in.readMapOfStrings();
        long bytes = in.getFilePointer() - position;
        assert bytes > 0;
        addBytesRead(bytes);
        return result;
    }

    @Override
    public Set<String> readSetOfStrings() throws IOException {
        long position = in.getFilePointer();
        Set<String> result = in.readSetOfStrings();
        long bytes = in.getFilePointer() - position;
        assert bytes > 0;
        addBytesRead(bytes);
        return result;
    }

    private static class RandomAccessIndexInput extends StoreMetricsIndexInput
        implements
            RandomAccessInput,
            MemorySegmentAccessInputAccess {
        private final RandomAccessInput delegate;

        private RandomAccessIndexInput(
            String resourceDescription,
            IndexInput in,
            PluggableDirectoryMetricsHolder<StoreMetrics> metricHolder
        ) {
            super(resourceDescription, in, metricHolder);
            assert in instanceof RandomAccessInput;
            this.delegate = (RandomAccessInput) in;
        }

        @Override
        public MemorySegmentAccessInput get() {
            return delegate instanceof MemorySegmentAccessInput ms ? ms : null;
        }

        @Override
        public long length() {
            return delegate.length();
        }

        @Override
        public byte readByte(long pos) throws IOException {
            byte result = delegate.readByte(pos);
            addBytesRead(1);
            return result;
        }

        @Override
        public short readShort(long pos) throws IOException {
            short result = delegate.readShort(pos);
            addBytesRead(2);
            return result;
        }

        @Override
        public int readInt(long pos) throws IOException {
            int result = delegate.readInt(pos);
            addBytesRead(4);
            return result;
        }

        @Override
        public long readLong(long pos) throws IOException {
            long result = delegate.readLong(pos);
            addBytesRead(8);
            return result;
        }

        @Override
        public void readBytes(long pos, byte[] bytes, int offset, int length) throws IOException {
            delegate.readBytes(pos, bytes, offset, length);
            addBytesRead(length);
        }
    }

    private static class MetricsRandomAccessInput implements RandomAccessInput {
        private final PluggableDirectoryMetricsHolder<StoreMetrics> metricHolder;
        private final RandomAccessInput delegate;

        private MetricsRandomAccessInput(RandomAccessInput delegate, PluggableDirectoryMetricsHolder<StoreMetrics> metricHolder) {
            this.delegate = delegate;
            this.metricHolder = metricHolder;
        }

        @Override
        public long length() {
            return delegate.length();
        }

        @Override
        public byte readByte(long pos) throws IOException {
            byte result = delegate.readByte(pos);
            metricHolder.instance().addBytesRead(1);
            return result;
        }

        @Override
        public short readShort(long pos) throws IOException {
            short result = delegate.readShort(pos);
            metricHolder.instance().addBytesRead(2);
            return result;
        }

        @Override
        public int readInt(long pos) throws IOException {
            int result = delegate.readInt(pos);
            metricHolder.instance().addBytesRead(4);
            return result;
        }

        @Override
        public long readLong(long pos) throws IOException {
            long result = delegate.readLong(pos);
            metricHolder.instance().addBytesRead(8);
            return result;
        }

        @Override
        public void readBytes(long pos, byte[] bytes, int offset, int length) throws IOException {
            delegate.readBytes(pos, bytes, offset, length);
            metricHolder.instance().addBytesRead(length);
        }
    }

    /**
     * Forwards {@link TieredPrefetchInput} to a delegate that implements it. Only chosen by {@link #wrap} when the delegate
     * has the capability, so callers never see a tiered input over local storage.
     */
    private static final class TieredStoreMetricsIndexInput extends StoreMetricsIndexInput implements TieredPrefetchInput {

        private TieredStoreMetricsIndexInput(
            String resourceDescription,
            IndexInput in,
            PluggableDirectoryMetricsHolder<StoreMetrics> metricHolder
        ) {
            super(resourceDescription, in, metricHolder);
            assert in instanceof TieredPrefetchInput;
        }

        @Override
        public Outcome ensureResident(long offset, long length) throws IOException {
            return ((TieredPrefetchInput) in).ensureResident(offset, length);
        }

        @Override
        public void ensureResident(long[] offsets, int length, int count, Outcome[] outcomes) throws IOException {
            ((TieredPrefetchInput) in).ensureResident(offsets, length, count, outcomes);
        }

        @Override
        public long residencyRegionSize() {
            return ((TieredPrefetchInput) in).residencyRegionSize();
        }
    }

    /**
     * The random-access counterpart of {@link TieredStoreMetricsIndexInput}, for delegates that are both.
     */
    private static final class TieredRandomAccessIndexInput extends RandomAccessIndexInput implements TieredPrefetchInput {

        private TieredRandomAccessIndexInput(
            String resourceDescription,
            IndexInput in,
            PluggableDirectoryMetricsHolder<StoreMetrics> metricHolder
        ) {
            super(resourceDescription, in, metricHolder);
            assert in instanceof TieredPrefetchInput;
        }

        @Override
        public Outcome ensureResident(long offset, long length) throws IOException {
            return ((TieredPrefetchInput) in).ensureResident(offset, length);
        }

        @Override
        public void ensureResident(long[] offsets, int length, int count, Outcome[] outcomes) throws IOException {
            ((TieredPrefetchInput) in).ensureResident(offsets, length, count, outcomes);
        }

        @Override
        public long residencyRegionSize() {
            return ((TieredPrefetchInput) in).residencyRegionSize();
        }
    }
}
