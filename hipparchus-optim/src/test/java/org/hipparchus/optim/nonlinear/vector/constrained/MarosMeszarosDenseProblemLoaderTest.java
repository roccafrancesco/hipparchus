/*
 * Licensed to the Hipparchus project under one or more
 * contributor license agreements.  See the NOTICE file distributed with
 * this work for additional information regarding copyright ownership.
 * The Hipparchus project licenses this file to You under the Apache License, Version 2.0
 * (the "License"); you may not use this file except in compliance with
 * the License.  You may obtain a copy of the License at
 *
 *      https://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
package org.hipparchus.optim.nonlinear.vector.constrained;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;

import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;

class MarosMeszarosDenseProblemLoaderTest {

    @TempDir
    Path tempDir;

    @Test
    void testObjNameSelectsObjectiveRow() throws IOException {
        final Path file = writeQps("objname.qps",
                "NAME          OBJNAME\n" +
                "OBJNAME\n" +
                "  COST\n" +
                "ROWS\n" +
                " N  FREE\n" +
                " N  COST\n" +
                " E  EQ1\n" +
                "COLUMNS\n" +
                "    X1        FREE       99.0    COST       2.0\n" +
                "    X1        EQ1         1.0\n" +
                "RHS\n" +
                "    RHS1      EQ1         3.0\n" +
                "BOUNDS\n" +
                " FR BND       X1\n" +
                "ENDATA\n");

        final MarosMeszarosDenseProblemLoader.DenseQPProblem problem =
                new MarosMeszarosDenseProblemLoader().loadDenseProblem(file);

        assertArrayEquals(new double[] { 2.0 }, problem.getC(), 0.0);
        assertArrayEquals(new double[] { 3.0 }, problem.getBeq(), 0.0);
        assertEquals(0, problem.getAiq().length);
    }

    @Test
    void testIntegerBoundTypesLoadedAsContinuousBounds() throws IOException {
        final Path file = writeQps("bounds.qps",
                "NAME          BOUNDS\n" +
                "ROWS\n" +
                " N  OBJ\n" +
                "COLUMNS\n" +
                "    X1        OBJ        1.0\n" +
                "    X2        OBJ        1.0\n" +
                "    X3        OBJ        1.0\n" +
                "BOUNDS\n" +
                " BV BND       X1\n" +
                " LI BND       X2        -2.0\n" +
                " UI BND       X3         5.0\n" +
                "ENDATA\n");

        final MarosMeszarosDenseProblemLoader.DenseQPProblem problem =
                new MarosMeszarosDenseProblemLoader().loadDenseProblem(file);

        assertArrayEquals(new double[] { 1.0, 1.0, 1.0 }, problem.getC(), 0.0);
        assertArrayEquals(new double[] { 1.0, 0.0, 0.0 }, problem.getAiq()[0], 0.0);
        assertArrayEquals(new double[] { -1.0, 0.0, 0.0 }, problem.getAiq()[1], 0.0);
        assertArrayEquals(new double[] { 0.0, 1.0, 0.0 }, problem.getAiq()[2], 0.0);
        assertArrayEquals(new double[] { 0.0, 0.0, 1.0 }, problem.getAiq()[3], 0.0);
        assertArrayEquals(new double[] { 0.0, 0.0, -1.0 }, problem.getAiq()[4], 0.0);
        assertArrayEquals(new double[] { 0.0, -1.0, -2.0, 0.0, -5.0 }, problem.getBiq(), 0.0);
    }

    private Path writeQps(final String name, final String content) throws IOException {
        final Path file = tempDir.resolve(name);
        Files.writeString(file, content);
        return file;
    }

}
