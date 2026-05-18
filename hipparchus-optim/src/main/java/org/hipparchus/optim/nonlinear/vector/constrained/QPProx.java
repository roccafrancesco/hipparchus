package org.hipparchus.optim.nonlinear.vector.constrained;

import java.util.ArrayList;
import java.util.List;

import org.hipparchus.linear.ArrayRealVector;
import org.hipparchus.linear.CholeskyDecomposition;
import org.hipparchus.linear.DecompositionSolver;
import org.hipparchus.linear.MatrixUtils;
import org.hipparchus.linear.RealMatrix;
import org.hipparchus.linear.RealVector;
import org.hipparchus.optim.OptimizationData;
import org.hipparchus.optim.nonlinear.scalar.ObjectiveFunction;
import org.hipparchus.util.FastMath;
import org.hipparchus.util.Precision;

/** Proximal-point outer loop around {@link QPDualActiveSolverW}. */
public class QPProx extends QPOptimizer {

    private static final double MU = 1.0e-4;
    private static final int MAX_OUTER = 30;
    private static final double EPS = FastMath.ulp(1.0);

    private QuadraticFunction function;
    private LinearEqualityConstraint eqConstraints;
    private LinearInequalityConstraint iqConstraints;
    private LinearBoundedConstraint bConstraints;
    private MatrixDecompositionTolerance matrixDecompositionTolerance;
    private QPMatrixMode matrixMode = QPMatrixMode.FULL;

    @Override
    protected void parseOptimizationData(final OptimizationData... optData) {
        super.parseOptimizationData(optData);
        this.function = null;
        this.eqConstraints = null;
        this.iqConstraints = null;
        this.bConstraints = null;
        this.matrixDecompositionTolerance = new MatrixDecompositionTolerance(EPS);
        this.matrixMode = QPMatrixMode.FULL;
        for (OptimizationData data : optData) {
            if (data instanceof ObjectiveFunction) {
                function = (QuadraticFunction) ((ObjectiveFunction) data).getObjectiveFunction();
            } else if (data instanceof LinearEqualityConstraint) {
                eqConstraints = (LinearEqualityConstraint) data;
            } else if (data instanceof LinearInequalityConstraint) {
                iqConstraints = (LinearInequalityConstraint) data;
            } else if (data instanceof LinearBoundedConstraint) {
                bConstraints = (LinearBoundedConstraint) data;
            } else if (data instanceof QPMatrixMode) {
                matrixMode = (QPMatrixMode) data;
            } else if (data instanceof MatrixDecompositionTolerance) {
                matrixDecompositionTolerance = (MatrixDecompositionTolerance) data;
            }
        }
    }

    @Override
    public LagrangeSolution doOptimize() {
        final RealMatrix gBase = matrixMode == QPMatrixMode.FULL ? function.getP() : function.getP().multiplyTransposed(function.getP());
        final RealVector gradBase = function.getQ();
        final double cst = function.getD();

        final QPDualActiveSolverW inner = new QPDualActiveSolverW();
        LagrangeSolution first = inner.optimize(new ObjectiveFunction(new QuadraticFunction(gBase, gradBase, cst)),
                                                eqConstraints, iqConstraints, bConstraints, matrixMode, matrixDecompositionTolerance);
        if (!isRetryCase(first)) {
            return first;
        }

        final int n = gBase.getColumnDimension();
        final int p = eqConstraints != null ? eqConstraints.dimY() : 0;
        final int m1 = iqConstraints != null ? iqConstraints.dimY() : 0;
        final int b1 = bConstraints != null ? bConstraints.dimY() : 0;
        final int m = m1 + 2 * b1;
        final int mc = p + m;

        final ConstraintPack pack = buildConstraints(n, p, m1, b1);
        final RealMatrix gProx = gBase.add(MatrixUtils.createRealIdentityMatrix(n).scalarMultiply(MU));

        RealVector x = new ArrayRealVector(n);
        RealVector prev = x.copy();
        RealVector u = new ArrayRealVector(0, 0);
        List<Integer> active = new ArrayList<>();

        final RealMatrix L = new CholeskyDecomposition(gProx).getL();
        QRUpdaterR qr = new QRUpdaterR(inverseLowerTriangular(L));

        for (int k = 0; k < MAX_OUTER; ++k) {
            final RealVector qk = gradBase.subtract(x.mapMultiply(MU));
            final LagrangeSolution sol = inner.optimizeWarmStart(x, u, active, gProx, qk, cst, p, m, pack.C, pack.c0, pack.weights, qr);
            final QPDualActiveSolverW.WarmState ws = inner.getLastWarmState();
            if (isFailedSolve(sol)) {
                /*
                 * Proximal-point continuation:
                 * keep partial warm state and continue outer iterations instead
                 * of stopping at the first inner failure.
                 */
                if (ws != null) {
                    if (ws.getX() != null && ws.getX().getDimension() == n) {
                        x = ws.getX().copy();
                    }
                    u = ws.getU() != null ? ws.getU().copy() : new ArrayRealVector(0, 0);
                    active = ws.getActive() != null ? new ArrayList<>(ws.getActive()) : new ArrayList<>();
                    if (ws.getQrUpdater() != null) {
                        qr = ws.getQrUpdater();
                    }
                    continue;
                }
                return sol;
            }
            if (ws != null) {
                prev = x.copy();
                x = ws.getX().copy();
                u = ws.getU().copy();
                active = new ArrayList<>(ws.getActive());
                qr = ws.getQrUpdater();
            } else {
                prev = x.copy();
                x = sol.getX().copy();
                u = extractActiveMultipliers(sol.getLambda(), active);
            }
            final double infNorm = x.subtract(prev).getLInfNorm();
            if (infNorm <= FastMath.sqrt(EPS)) {
                return sol;
            }
        }

        return new LagrangeSolution(new ArrayRealVector(0), new ArrayRealVector(1, QPDualActiveSolverW.ERROR_INFEASIBLE), 0.0);
    }

    private boolean isRetryCase(final LagrangeSolution s) {
        return isFailedSolve(s);
    }

    private boolean isFailedSolve(final LagrangeSolution s) {
        return s == null || s.getX() == null || s.getX().getDimension() == 0;
    }

    private RealVector extractActiveMultipliers(final RealVector lambda, final List<Integer> active) {
        final RealVector u = new ArrayRealVector(active.size());
        for (int i = 0; i < active.size(); ++i) {
            u.setEntry(i, lambda.getEntry(active.get(i)));
        }
        return u;
    }

    private RealMatrix inverseLowerTriangular(final RealMatrix L) {
        final int n = L.getRowDimension();
        final RealMatrix Linv = MatrixUtils.createRealMatrix(n, n);
        for (int i = 0; i < n; i++) {
            final RealVector e = new ArrayRealVector(n);
            e.setEntry(i, 1.0);
            MatrixUtils.solveLowerTriangularSystem(L, e);
            Linv.setColumnVector(i, e);
        }
        return Linv;
    }

    private ConstraintPack buildConstraints(final int n, final int p, final int m1, final int b1) {
        final int mc = p + m1 + 2 * b1;
        final RealMatrix C = MatrixUtils.createRealMatrix(n, mc);
        final double[] c0Data = new double[mc];
        final double[] wData = new double[mc];

        if (p > 0) {
            final RealMatrix Ae = eqConstraints.getA();
            final RealVector be = eqConstraints.getLowerBound();
            for (int j = 0; j < p; j++) {
                c0Data[j] = -be.getEntry(j);
                double sum = 0;
                for (int i = 0; i < n; i++) {
                    double v = Ae.getEntry(j, i);
                    C.setEntry(i, j, v);
                    sum += v * v;
                }
                wData[j] = sum > Precision.SAFE_MIN ? 1.0 / FastMath.sqrt(sum) : 1.0;
            }
        }
        if (m1 > 0) {
            final RealMatrix Ai = iqConstraints.jacobian(null);
            final RealVector bi = iqConstraints.getLowerBound();
            for (int j = 0; j < m1; j++) {
                c0Data[p + j] = -bi.getEntry(j);
                double sum = 0;
                for (int i = 0; i < n; i++) {
                    double v = Ai.getEntry(j, i);
                    C.setEntry(i, p + j, v);
                    sum += v * v;
                }
                wData[p + j] = sum > Precision.SAFE_MIN ? 1.0 / FastMath.sqrt(sum) : 1.0;
            }
        }
        if (b1 > 0) {
            final RealMatrix Ab = bConstraints.jacobian(null);
            final RealVector lb = bConstraints.getLowerBound();
            final RealVector ub = bConstraints.getUpperBound();
            for (int j = 0; j < b1; j++) {
                c0Data[p + m1 + j] = -lb.getEntry(j);
                c0Data[p + m1 + b1 + j] = ub.getEntry(j);
                double sum = 0;
                for (int i = 0; i < n; i++) {
                    double v = Ab.getEntry(j, i);
                    C.setEntry(i, p + m1 + j, v);
                    C.setEntry(i, p + m1 + b1 + j, -v);
                    sum += v * v;
                }
                double w = sum > Precision.SAFE_MIN ? 1.0 / FastMath.sqrt(sum) : 1.0;
                wData[p + m1 + j] = w;
                wData[p + m1 + b1 + j] = w;
            }
        }
        return new ConstraintPack(C, new ArrayRealVector(c0Data, false), new ArrayRealVector(wData, false));
    }

    private static final class ConstraintPack {
        private final RealMatrix C;
        private final RealVector c0;
        private final RealVector weights;

        private ConstraintPack(final RealMatrix C, final RealVector c0, final RealVector weights) {
            this.C = C;
            this.c0 = c0;
            this.weights = weights;
        }
    }
}
