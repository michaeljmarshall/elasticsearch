/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the "Elastic License
 * 2.0", the "GNU Affero General Public License v3.0 only", and the "Server Side
 * Public License v 1"; you may not use this file except in compliance with, at
 * your election, the "Elastic License 2.0", the "GNU Affero General Public
 * License v3.0 only", or the "Server Side Public License, v 1".
 */

package org.elasticsearch.search.profile.query;

import org.elasticsearch.common.io.stream.Writeable;
import org.elasticsearch.core.TieredPrefetchInput;
import org.elasticsearch.test.AbstractXContentSerializingTestCase;
import org.elasticsearch.xcontent.XContentParser;

import java.io.IOException;

import static org.hamcrest.Matchers.equalTo;

public class TieredPrefetchOutcomeCountsTests extends AbstractXContentSerializingTestCase<TieredPrefetchOutcomeCounts> {

    public static TieredPrefetchOutcomeCounts createTestItem() {
        return new TieredPrefetchOutcomeCounts(randomNonNegativeLong(), randomNonNegativeLong(), randomNonNegativeLong());
    }

    @Override
    protected TieredPrefetchOutcomeCounts createTestInstance() {
        return createTestItem();
    }

    @Override
    protected TieredPrefetchOutcomeCounts mutateInstance(TieredPrefetchOutcomeCounts instance) {
        long resident = instance.resident();
        long fetching = instance.fetching();
        long skipped = instance.skipped();
        switch (between(0, 2)) {
            case 0 -> resident = randomValueOtherThan(resident, () -> randomNonNegativeLong());
            case 1 -> fetching = randomValueOtherThan(fetching, () -> randomNonNegativeLong());
            case 2 -> skipped = randomValueOtherThan(skipped, () -> randomNonNegativeLong());
            default -> throw new AssertionError("unreachable");
        }
        return new TieredPrefetchOutcomeCounts(resident, fetching, skipped);
    }

    @Override
    protected TieredPrefetchOutcomeCounts doParseInstance(XContentParser parser) throws IOException {
        parser.nextToken();
        return TieredPrefetchOutcomeCounts.fromXContent(parser);
    }

    @Override
    protected Writeable.Reader<TieredPrefetchOutcomeCounts> instanceReader() {
        return TieredPrefetchOutcomeCounts::new;
    }

    /**
     * The accumulator must tally each outcome under its own counter, and a copy must stop tracking further additions,
     * since the profiler stores the copy while the query may keep accumulating.
     */
    public void testAccumulateAndCopy() {
        TieredPrefetchOutcomeCounts counts = new TieredPrefetchOutcomeCounts();
        assertTrue(counts.isEmpty());
        int resident = between(0, 10);
        int fetching = between(0, 10);
        int skipped = between(0, 10);
        for (int i = 0; i < resident; i++) {
            counts.add(TieredPrefetchInput.Outcome.RESIDENT);
        }
        for (int i = 0; i < fetching; i++) {
            counts.add(TieredPrefetchInput.Outcome.FETCHING);
        }
        for (int i = 0; i < skipped; i++) {
            counts.add(TieredPrefetchInput.Outcome.SKIPPED);
        }
        assertThat(counts, equalTo(new TieredPrefetchOutcomeCounts(resident, fetching, skipped)));
        assertThat(counts.total(), equalTo((long) resident + fetching + skipped));
        assertThat(counts.isEmpty(), equalTo(resident + fetching + skipped == 0));

        TieredPrefetchOutcomeCounts copy = counts.copy();
        counts.add(TieredPrefetchInput.Outcome.FETCHING);
        assertThat(copy, equalTo(new TieredPrefetchOutcomeCounts(resident, fetching, skipped)));
        assertThat(counts.fetching(), equalTo((long) fetching + 1));

        TieredPrefetchOutcomeCounts sum = new TieredPrefetchOutcomeCounts();
        sum.add(counts);
        sum.add(copy);
        assertThat(sum, equalTo(new TieredPrefetchOutcomeCounts(2L * resident, 2L * fetching + 1, 2L * skipped)));
    }
}
