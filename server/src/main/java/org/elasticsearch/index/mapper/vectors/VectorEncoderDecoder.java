/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the "Elastic License
 * 2.0", the "GNU Affero General Public License v3.0 only", and the "Server Side
 * Public License v 1"; you may not use this file except in compliance with, at
 * your election, the "Elastic License 2.0", the "GNU Affero General Public
 * License v3.0 only", or the "Server Side Public License, v 1".
 */

package org.elasticsearch.index.mapper.vectors;

import org.apache.lucene.util.BitUtil;
import org.apache.lucene.util.BytesRef;
import org.elasticsearch.index.IndexVersion;
import org.elasticsearch.index.codec.vectors.BFloat16;
import org.elasticsearch.simdvec.ESVectorUtil;

import java.lang.invoke.VarHandle;
import java.nio.ByteOrder;

import static org.elasticsearch.index.mapper.vectors.DenseVectorFieldMapper.LITTLE_ENDIAN_FLOAT_STORED_INDEX_VERSION;
import static org.elasticsearch.index.mapper.vectors.DenseVectorFieldMapper.MAGNITUDE_STORED_INDEX_VERSION;

public final class VectorEncoderDecoder {
    public static final byte INT_BYTES = 4;

    private VectorEncoderDecoder() {}

    public static int denseVectorLength(IndexVersion indexVersion, BytesRef vectorBR) {
        return indexVersion.onOrAfter(MAGNITUDE_STORED_INDEX_VERSION)
            ? (vectorBR.length - INT_BYTES) / INT_BYTES
            : vectorBR.length / INT_BYTES;
    }

    /**
     * Decodes the last 4 bytes of the encoded vector, which contains the vector magnitude.
     * NOTE: this function can only be called on vectors from an index version greater than or
     * equal to 7.5.0, since vectors created prior to that do not store the magnitude.
     */
    public static float decodeMagnitude(IndexVersion indexVersion, BytesRef vectorBR) {
        assert indexVersion.onOrAfter(MAGNITUDE_STORED_INDEX_VERSION);
        int offset = vectorBR.offset + vectorBR.length - INT_BYTES;
        return indexVersion.onOrAfter(LITTLE_ENDIAN_FLOAT_STORED_INDEX_VERSION)
            ? (float) BitUtil.VH_LE_FLOAT.get(vectorBR.bytes, offset)
            : (float) BitUtil.VH_BE_FLOAT.get(vectorBR.bytes, offset);
    }

    /**
     * Calculates vector magnitude
     */
    private static float calculateMagnitude(float[] decodedVector) {
        return (float) Math.sqrt(ESVectorUtil.dotProduct(decodedVector, decodedVector));
    }

    public static float getMagnitude(IndexVersion indexVersion, BytesRef vectorBR, float[] decodedVector) {
        if (vectorBR == null) {
            throw new IllegalArgumentException(DenseVectorScriptDocValues.MISSING_VECTOR_FIELD_MESSAGE);
        }
        if (indexVersion.onOrAfter(MAGNITUDE_STORED_INDEX_VERSION)) {
            return decodeMagnitude(indexVersion, vectorBR);
        } else {
            return calculateMagnitude(decodedVector);
        }
    }

    /**
     * Decodes a BytesRef into the provided array of floats
     * @param vectorBR - dense vector encoded in BytesRef
     * @param vector - array of floats where the decoded vector should be stored
     */
    public static void decodeDenseVector(IndexVersion indexVersion, BytesRef vectorBR, float[] vector) {
        if (vectorBR == null) {
            throw new IllegalArgumentException(DenseVectorScriptDocValues.MISSING_VECTOR_FIELD_MESSAGE);
        }
        VarHandle vh = indexVersion.onOrAfter(LITTLE_ENDIAN_FLOAT_STORED_INDEX_VERSION) ? BitUtil.VH_LE_FLOAT : BitUtil.VH_BE_FLOAT;
        for (int i = 0; i < vector.length; i++) {
            vector[i] = (float) vh.get(vectorBR.bytes, vectorBR.offset + i * Float.BYTES);
        }
    }

    public static void decodeBFloat16DenseVector(BytesRef vectorBR, float[] vector) {
        if (vectorBR == null) {
            throw new IllegalArgumentException(DenseVectorScriptDocValues.MISSING_VECTOR_FIELD_MESSAGE);
        }
        BFloat16.bFloat16ToFloat(vectorBR.bytes, vectorBR.offset, vector, 0, vector.length, ByteOrder.LITTLE_ENDIAN);
    }

    /**
     * Decodes a BytesRef into the provided array of bytes
     * @param vectorBR - dense vector encoded in BytesRef
     * @param vector - array of bytes where the decoded vector should be stored
     */
    public static void decodeDenseVector(IndexVersion indexVersion, BytesRef vectorBR, byte[] vector) {
        if (vectorBR == null) {
            throw new IllegalArgumentException(DenseVectorScriptDocValues.MISSING_VECTOR_FIELD_MESSAGE);
        }
        if (indexVersion.onOrAfter(LITTLE_ENDIAN_FLOAT_STORED_INDEX_VERSION)) {
            ByteBuffer fb = ByteBuffer.wrap(vectorBR.bytes, vectorBR.offset, vectorBR.length).order(ByteOrder.LITTLE_ENDIAN);
            fb.get(vector);
        } else {
            ByteBuffer byteBuffer = ByteBuffer.wrap(vectorBR.bytes, vectorBR.offset, vectorBR.length);
            for (int dim = 0; dim < vector.length; dim++) {
                vector[dim] = byteBuffer.get(dim * vectorBR.offset);
            }
        }
    }

    public static float[] getMultiMagnitudes(BytesRef magnitudes) {
        assert magnitudes.length % Float.BYTES == 0;
        float[] multiMagnitudes = new float[magnitudes.length / Float.BYTES];
        for (int i = 0; i < multiMagnitudes.length; i++) {
            multiMagnitudes[i] = (float) BitUtil.VH_LE_FLOAT.get(magnitudes.bytes, magnitudes.offset + i * Float.BYTES);
        }
        return multiMagnitudes;
    }

}
