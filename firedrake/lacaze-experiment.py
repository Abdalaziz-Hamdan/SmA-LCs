# -*- coding: utf-8 -*-
from firedrake import *
from defcon import *
from petsc4py import PETSc
import numpy as np
make_initial_guess = False

class SmecticProblem(BifurcationProblem):
    def mesh(self, comm):

        # The vertical height of the domain is 160 nm. We rescale this to be 1 computational unit
        # The horizonal length is 2 * 113 nm
        nrefine = 2
        X_refine = 12
        assert X_refine % 2 == 0
        Y_refine = X_refine//3 * 2
        mesh = PeriodicRectangleMesh(X_refine*(nrefine+1), Y_refine*(nrefine+1), 226/160, 160/160, diagonal="crossed", direction="x", comm=comm)

        # Deform the height of the domain.
        # We want the curve of the top surface to interpolate
        # the data points (0, 160), (56.5, 144), (113, 90)
        # which we must then rescale and mirror.
        # The resulting interpolating polynomial for the _height_ of the left-half domain is
        # -0.952306 x^2 + 0.0530973 x + 1.
        # So the formula for the displacement is -0.952306 x^2 + 0.0530973 x.

        (x, y) = SpatialCoordinate(mesh)
        top_displacement = conditional(le(x, 113/160),  # mirroring in the x = 113/160 line
                                       -0.952306 * x**2 + 0.0530973 * x,
                                       -0.952306 * (226/160 - x)**2 + 0.0530973 * (226/160 - x))

        # Solve linear elasticity to get the deformed mesh
        mesh_degree = 1  # want to use CG2 but can't because of bugs
        V = VectorFunctionSpace(mesh, "CG", mesh_degree)
        u = Function(V)

        mu = Constant(1e-3)
        lmbda = Constant(1)
        epsilon = lambda u: 0.5*(grad(u) + grad(u).T)
        Id = Identity(2)
        sigma = lambda u: lmbda*div(u)*Id + 2*mu*epsilon(u)
        F = inner(sigma(u), epsilon(TestFunction(V)))*dx
        bcs = [DirichletBC(V, 0, 1),  # clamp the bottom
               DirichletBC(V.sub(0), 0, 2),  # don't slide along the top
               DirichletBC(V.sub(1), top_displacement, 2)]
        sp = {"ksp_type": "preonly",
              "pc_type": "cholesky",
              "pc_factor_mat_solver_type": "mumps"}
        solve(F == 0, u, bcs=bcs, solver_parameters=sp)

        X = interpolate(SpatialCoordinate(mesh) + u, VectorFunctionSpace(mesh, "DG", mesh_degree))
        newmesh = Mesh(X)

        return newmesh

    def function_space(self, mesh):
        self.density_degree = 5
        self.Q_degree = 5
        U = FunctionSpace(mesh, "Argyris", self.density_degree)
        V = VectorFunctionSpace(mesh, "CG", self.Q_degree, dim=2)
        Z  = MixedFunctionSpace([U, V])
        self.CG = FunctionSpace(mesh, "CG", self.density_degree)

        print("Z.dim(): %s %s" % (Z.dim(), [Z.sub(i).dim() for i in range(2)]))

        return Z

    def parameters(self):
        q = Constant(0)
        W = Constant(0)
        return [(q, "q", r"$q$"),
                (W, "W", "anchorweight")]

    def energy(self, z, params):
        q = params[0]
        q0 = Constant(30)
        scale = q/q0
        print(f"Scaling factor: {float(scale)}")
        # Scott has nagging doubts here about the scaling
        W = params[1]      * scale**2
        a = Constant(-5*2) * scale**3
        b = Constant(0)    * scale**3
        c = Constant(5*2)  * scale**3
        B = Constant(1e-5) * (scale**(-1))
        K = Constant(0.6)  * scale
        l = Constant(1)    * scale**3

        s = FacetNormal(z.function_space().mesh())

        (u, d) = split(z)
        Q = as_tensor([[d[0], d[1]],
                       [d[1], -d[0]]])
        Q_bottom = as_tensor([[1/2, 0], [0, -1/2]])
        I = Identity(2)
        Q_top = outer(s, s) - 1/2 * I
        mat = grad(grad(u)) + q**2 * (Q+I/2) * u
        E = (
            + a/2 * u**2 * dx
            + b/3 * u**3 * dx
            + c/4 * u**4 * dx
            + B   * inner(mat, mat) * dx
            + K/2 * inner(grad(Q), grad(Q)) * dx
            - l * tr(Q*Q) * dx
            + l * dot(tr(Q*Q), tr(Q*Q)) * dx
            + W/2 * inner(Q-Q_bottom, Q-Q_bottom) * ds(1)
            + W/2 * inner(Q-Q_top, Q-Q_top) * ds(2)
            + Constant(1e4)*inner(u - 1, u - 1)*ds(2)  # Should this be more complicated?
            )

        return E

    def lagrangian(self, z, params):
        E = self.energy(z, params)
        return E

    def residual(self, z, params, w):
        L = self.lagrangian(z, params)
        F = derivative(L, z, w)
        return F

    def boundary_conditions(self, Z, params):
        #return [DirichletBC(Z.sub(0), 1, 2)]
        return []

    def functionals(self):
        def energy(z, params):
            return assemble(self.energy(z, params))

        return [
                (energy, "energy", r"$E(u, Q)$"),
               ]

    def number_initial_guesses(self, params):
        return 1

    def initial_guess_zero(self, Z, params, n):
        return Function(Z)

    def initial_guess(self, Z, params, n):
        lu = {"ksp_type": "preonly",
              "pc_type": "lu",
              "mat_type": "aij",
              "pc_factor_mat_solver_type": "mumps",
              "mat_mumps_icntl_14": 200,}

        if make_initial_guess:
            print("Generating initial guess")
            # Tim's form for TFCD, this finds something for nrefine=0 or 1
            (x, y) = SpatialCoordinate(Z.mesh())
            R = Constant(1.0)
            denomy = sqrt(R**2+x**2-2*R*sqrt(x**2)+y**2)+Constant(1e-10)
            denomx = (sqrt(x**2)+Constant(1e-10)) * denomy
            nx = x*(sqrt(x**2)-R)/denomy
            ny = y/denomy
            q0 = conditional(x**2>R**2, as_vector([-1/2, 0]),
                    as_vector([nx**2-1/2, nx*ny]))

            z = Function(Z)
            z.sub(0).project(Constant(1.0), solver_parameters=lu)
            z.sub(1).interpolate(q0)
        else:
            print("Reading initial guess from input/q-35_0.vtu")
            import vtktools
            vtu = vtktools.vtu("input/q-35_0.vtu")
            mesh = Z.mesh()

            density_coord_space = VectorFunctionSpace(mesh, "CG", self.density_degree)
            X = interpolate(SpatialCoordinate(mesh), density_coord_space)
            reader = lambda X: vtu.ProbeData(np.c_[X, np.zeros(X.shape[0])], "Density")[:,0]
            u = Function(self.CG)
            u.dat.data[:] = reader(X.dat.data_ro)

            Q_coord_space = VectorFunctionSpace(mesh, "CG", self.Q_degree)
            X = interpolate(SpatialCoordinate(mesh), Q_coord_space)
            reader = lambda X: vtu.ProbeData(np.c_[X, np.zeros(X.shape[0])], "ComponentsOfQ")[:,0:2]
            d = Function(Z.sub(1))
            d.dat.data[:] = reader(X.dat.data_ro)

            z = Function(Z)
            z.sub(0).project(u, solver_parameters=lu)
            z.sub(1).assign(d)

        return z

    def number_solutions(self, params):
        if make_initial_guess:
            return 1
        return float("inf")

    def solver_parameters(self, params, task, **kwargs):
        if isinstance(task, DeflationTask):
            damping = 0.9
            maxits = 500
        else:
            damping = 1.0
            maxits = 500

        params = {
            "snes_max_it": maxits,
            "snes_atol": 1.0e-6,
            "snes_rtol": 1.0e-8,
            "snes_divergence_tolerance": 1.0e10,
            "snes_monitor": None,
            "snes_linesearch_type": "l2",
            "snes_linesearch_monitor": None,
            "snes_linesearch_maxstep": 1.0,
            "snes_linesearch_damping": damping,
            "snes_converged_reason": None,
            "mat_type": "aij",
            "ksp_type": "preonly",
            "pc_type": "lu",
            "pc_factor_mat_solver_type": "mumps",
            "mat_mumps_icntl_14": 200,
            "mat_mumps_icntl_24": 1,
            "mat_mumps_icntl_13": 1,
            "tao_type": "bnls",
            "tao_gmonitor": None,
            #"tao_monitor": None,
            #"tao_ls_monitor": None,
            "tao_bnk_ksp_type": "gmres",
            "tao_bnk_ksp_converged_reason": None,
            "tao_bnk_ksp_monitor_true_residual": None,
            "tao_bnk_pc_type": "lu",
            "tao_ntr_pc_factor_mat_solver_type": "mumps",
            #"tao_ntr_pc_type": "lmvm",
            "tao_ls_type": "armijo",
            "tao_converged_reason": None,
        }
        return params

    def save_pvd(self, z, pvd, params):
        (u, d) = z.split()
        uv = project(u, self.CG)
        uv.rename("Density")

        # visualize the director
        mesh = z.function_space().mesh()
        d0 = d[0]
        d1 = d[1]
        Q11 = Function(FunctionSpace(mesh, "CG", self.Q_degree)).interpolate(d0)
        Q12 = Function(FunctionSpace(mesh, "CG", self.Q_degree)).interpolate(d1)
        Q11.rename("Q11")
        Q12.rename("Q12")
        Q = interpolate(as_tensor([[d0, d1], [d1, -d0]]), TensorFunctionSpace(mesh, "CG", self.Q_degree))
        Q.rename("Q")
        eigs, eigv = np.linalg.eigh(np.array(Q.vector()))
        s = Function(FunctionSpace(mesh, "CG", self.Q_degree)).interpolate(2*sqrt(dot(d,d)))
        s.rename("OrderParameter")
        s_eig = Function(FunctionSpace(mesh, "CG", self.Q_degree))
        s_eig.vector()[:] = 2*eigs[:,1]
        s_eig.rename("OrderParameterViaEig")
        n = Function(VectorFunctionSpace(mesh, "CG", self.Q_degree))
        n.vector()[:,:] = eigv[:,:,1]
        n.rename("Director")
        d.rename("ComponentsOfQ")

        pvd.write(uv, Q, n, s, s_eig, d)

    def monitor(self, params, branchid, solution, functionals):
        filename = "output/pvd/q-%s/solution-%d.pvd" % (params[0], branchid)
        pvd = File(filename, comm=solution.function_space().mesh().comm)
        self.save_pvd(solution, pvd, params)
        print("Wrote to %s" % filename)

    def compute_stability_disabled(self, params, branchid, z, hint=None):
        Z = z.function_space()
        trial = TrialFunction(Z)
        test  = TestFunction(Z)

        bcs = self.boundary_conditions(Z, params)
        comm = Z.mesh().mpi_comm()

        F = self.residual(z, [Constant(p) for p in params], test)
        J = derivative(F, z, trial)

        # Build the LHS matrix
        A = assemble(J, bcs=bcs, mat_type="aij")
        A = A.M.handle

        pc = PETSc.PC().create(comm)
        pc.setOperators(A)
        pc.setType("cholesky")
        pc.setFactorSolverType("mumps")
        pc.setUp()

        F = pc.getFactorMatrix()
        (neg, zero, pos) = F.getInertia()

        print("Inertia: (-: %s, 0: %s, +: %s)" % (neg, zero, pos))
        expected_dim = 0

        # Nocedal & Wright, theorem 16.3
        if neg == expected_dim:
            is_stable = True
        else:
            is_stable = False

        d = {"stable": (neg, zero, pos)}
        return d


    def predict_disabled(self, *args, **kwargs):
        return tangent(*args, **kwargs)


if __name__ == "__main__":
    dc = DeflatedContinuation(problem=SmecticProblem(), teamsize=4, verbose=True, profile=False, clear_output=True, logfiles=True)
    if make_initial_guess:
        params = linspace(30, 35, 6)
    else:
        params = linspace(30, 320, 581)
    dc.run(values={"q": params, "W": 10}, freeparam="q")
