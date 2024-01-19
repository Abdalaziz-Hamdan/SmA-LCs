# -*- coding: utf-8 -*-
from firedrake import *
from defcon import *
from petsc4py import PETSc
import numpy as np
make_initial_guess = True
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

import os
class SmecticProblem(BifurcationProblem):
    def mesh(self, comm):

        # The vertical height of the domain is 160 nm. We rescale this to be 1 computational unit
        # The horizonal length is 2 * 113 nm
        nrefine = 2
        X_refine = 36
        assert X_refine % 2 == 0
        Y_refine = X_refine//3 * 2
        mesh = RectangleMesh(X_refine*(nrefine+1), Y_refine*(nrefine+1), 226/160, 161.83333/160, diagonal="crossed", comm=comm)

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

#        top_displacement = conditional(le(x, 113/160),  # mirroring in the x = 113/160 line
#                                       conditional(x < 100/160, (30/160) + sqrt((130/160)**2 - x**2),
#                                       7.408360190*x**2 - 10.464308768*x + 4.3529662724) - 1,
#                                       conditional(x > 126/160, (30/160) + sqrt((130/160)**2 - (226/160-x)**2),
#                                       7.408360190*(226/160-x)**2 - 10.464308768*(226/160-x) + 4.3529662724) - 1)
        # Solve linear elasticity to get the deformed mesh
        mesh_degree = 3  # want to use CG2 but can't because of bugs
        V = VectorFunctionSpace(mesh, "CG", mesh_degree)
        u = Function(V)
        mu = 1e-3
        lame =  10
        J = (mu/2)*inner(2*sym(grad(u)),sym(grad(u)))*dx + ((lame)/2)*inner(div(u),div(u))*dx
        F = derivative(J,u)
        bcs = [
               DirichletBC(V, 0, 1),  # clamp the bottom
               DirichletBC(V, 0, 2),  # clamp the bottom
               DirichletBC(V, 0, 3),  # clamp the bottom
               DirichletBC(V.sub(0), 0, 4),  # don't slide along the top
               DirichletBC(V.sub(1), top_displacement, 4)]
        sp = {"ksp_type": "preonly",
              "pc_type": "cholesky",
              "pc_factor_mat_solver_type": "mumps"}
        solve(F == 0, u, bcs=bcs, solver_parameters=sp)
        X = interpolate(SpatialCoordinate(mesh) + u, VectorFunctionSpace(mesh, "CG", mesh_degree))
        newmesh = Mesh(X)

        return newmesh

    def function_space(self, mesh):
        self.density_degree = 4
        self.Q_degree = 3
        U = FunctionSpace(mesh, "CG", self.density_degree)
        V = VectorFunctionSpace(mesh, "CG", self.Q_degree, dim=2)
        Z  = MixedFunctionSpace([U, V])
        self.CG = FunctionSpace(mesh, "CG", self.density_degree)

        print("Z.dim(): %s %s" % (Z.dim(), [Z.sub(i).dim() for i in range(2)]))

        return Z

    def parameters(self):
        q = Constant(320.27564)
        W = Constant(10)
        d0 = Constant(3)
        ii = Constant(1)
        jj = Constant(1)
        return [(q, "q", r"$q$"),
                (W, "W", "anchorweight"),
                (d0, "d0", "dimenstionscaling"),
                (ii, "ii", "ii"),
                (jj,"jj", "jj"),
                                ]

    def energy(self, z, params):
        q = params[0]
        q0 = Constant(30)
        scale = q/q0
        print(f"Scaling factor: {float(scale)}")
        # Scott has nagging doubts here about the scaling
        d0 = params[2]
        ii = params[3]
        jj = params[4]
        W = params[1]      * scale**(d0-1)*100
        a = Constant(-5*2) * ii*scale**d0
        b = Constant(0)    * ii*scale**d0
        c = Constant(5*2)  * ii*scale**d0
        B = Constant(1e-5) * (scale**(d0-4))*jj
        K = Constant(0.6)  * scale**(d0-2)*100
        l = Constant(1)    * ii*scale**d0
        s = FacetNormal(z.function_space().mesh())

        (u, d) = split(z)
        Q = as_tensor([[d[0], d[1]],
                       [d[1], -d[0]]])
        (x, y) = SpatialCoordinate(z.function_space().mesh())
        Q_LR = conditional(y > 30/160, as_tensor([[-1/2, 0], [0, 1/2]]),
                    as_tensor([[1/2, 0], [0, -1/2]]))

        Q_bottom = as_tensor([[1/2, 0], [0, -1/2]])
        I = Identity(2)
        Q_top = outer(s, s) - 1/2 * I
        mat = grad(grad(u)) + q**2 * (Q+I/2) * u
        h = sqrt(avg(CellVolume(z.function_space().mesh())))
        E = (
            + a/2 * u**2 * dx
            + b/3 * u**3 * dx
            + c/4 * u**4 * dx
            + B   * inner(mat, mat) * dx
            + K/2 * inner(grad(Q), grad(Q)) * dx
            - l * tr(Q*Q) * dx
            + l * dot(tr(Q*Q), tr(Q*Q)) * dx
            + W/2 * inner(Q-Q_bottom, Q-Q_bottom) * ds(3)
            + W/2 * inner(Q-Q_top, Q-Q_top) * ds(4)
            + W/2 * inner(Q-Q_LR, Q-Q_LR) * ds(1)
            + W/2 * inner(Q-Q_LR, Q-Q_LR) * ds(2)
            + 20/h*inner(u - 1, u - 1)*ds(4)
            + 20/h*inner(jump(grad(u), s), jump(grad(u), s))*dS
             )

        return E

    def lagrangian(self, z, params):
        E = self.energy(z, params)
        return E

    def residual(self, z, params, w):
        q = params[0]
        d0 = params[2]
        q0 = Constant(30)
        scale = q/q0
        print(f"Scaling factor: {float(scale)}")
        # Scott has nagging doubts here about the scaling
        (u, d) = split(z)
        (phi, t) = split(w)
        L = self.lagrangian(z, params)
        F = (
           derivative(L, z, w)
             )
        return F

    def boundary_conditions(self, Z, params):
        return [
                 ]

    def functionals(self):

        def energy(z, params):
            return assemble(self.energy(z, params))

        def E_1(z, params):
            q = params[0]
            d0 = params[2]
            ii = params[3]
            q0 = Constant(30)
            scale = q/q0
            a = Constant(-5*2) * ii*scale**d0
            b = Constant(0)    * ii*scale**d0
            c = Constant(5*2)  * ii*scale**d0
            u = z.sub(0)
            return assemble(a/2 * u**2 * dx + b/3 * u**3 * dx + c/4 * u**4 * dx )

        def E_2(z, params):
            I = Identity(2)
            q0 = Constant(30)
            d0 = params[2]
            jj = params[4]
            q = params[0]
            scale = q/q0
            W = params[1]      * scale**(d0-1)*100
            u = z.sub(0)
            (u, d) = split(z)
            Q = as_tensor([[d[0], d[1]],
                       [d[1], -d[0]]])
            (x, y) = SpatialCoordinate(z.function_space().mesh())
            mat = grad(grad(u)) + q**2 * (Q+I/2) * u
            B = Constant(1e-5) * (scale**(d0-4))*jj
            return assemble( B * inner(mat, mat) * dx)

        def E_3(z, params):
            q = params[0]
            d0 = params[2]
            q0 = Constant(30)
            scale = q/q0
            K = Constant(0.6)  * scale**(d0-2)*100
            (u, d) = split(z)
            Q = as_tensor([[d[0], d[1]],
                       [d[1], -d[0]]])
            return assemble( K/2 * inner(grad(Q), grad(Q)) * dx
                            )
        def E_4(z, params):
            (u, d) = split(z)
            q = params[0]
            q0 = Constant(30)
            d0 = params[2]
            ii = params[3]
            scale = q/q0
            l = Constant(1)    *ii* scale**d0
            Q = as_tensor([[d[0], d[1]],
                       [d[1], -d[0]]])
            return assemble( - l * tr(Q*Q) * dx + l * dot(tr(Q*Q), tr(Q*Q)) * dx)

        def E_5(z, params):
            (u, d) = split(z)
            q = params[0]
            d0 = params[2]
            ii = params[3]
            q0 = Constant(30)
            scale = q/q0
            l = Constant(1)    *ii* scale**d0
            (x, y) = SpatialCoordinate(z.function_space().mesh())
            Q_bottom = as_tensor([[1/2, 0], [0, -1/2]])
            I = Identity(2)
            s = FacetNormal(z.function_space().mesh())
            Q_top = outer(s, s) - 1/2 * I
            Q = as_tensor([[d[0], d[1]],
                       [d[1], -d[0]]])
            W = params[1]      * scale**(d0-1)*100
            Q_LR = conditional(y > 30/160, as_tensor([[-1/2, 0], [0, 1/2]]),
                    as_tensor([[1/2, 0], [0, -1/2]]))

            return assemble( W/2 * inner(Q-Q_bottom, Q-Q_bottom) * ds(3)
                             + W/2 * inner(Q-Q_top, Q-Q_top) * ds(4)
                            + W/2 * inner(Q-Q_LR, Q-Q_LR) * ds(1)
                            + W/2 * inner(Q-Q_LR, Q-Q_LR) * ds(2)
                         )

        def E_6(z, params):
            (u, d) = split(z)
            h = sqrt(avg(CellVolume(z.function_space().mesh())))
            return assemble( 20/h*inner(u - 1, u - 1)*ds(4))

        return [
                (energy, "energy", r"$E(u, Q)$"),
                (E_1, "E_1", r"$E_1$"),
                (E_2, "E_2", r"$E_2$"),
                (E_3, "E_3", r"$E_3$"),
                (E_4, "E_4", r"$E_4$"),
                (E_5, "E_5", r"$E_5$"),
                (E_6, "E_6", r"$E_6$"),
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
            q = params[0]
            z.sub(0).project(Constant(1), solver_parameters=lu)
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
            return float("inf")
        return float("inf")

    def solver_parameters(self, params, task, **kwargs):
        if isinstance(task, DeflationTask):
            damping = 0.7
            maxits = 5000
        else:
            damping = .7
            maxits = 5000

        params = {
            "snes_max_it": maxits,
            "snes_atol": 1.0e-7,
            "snes_rtol": 1.0e-9,
            "snes_stol": 1.0e-100,
            "snes_divergence_tolerance": 1.0e16,
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


    def checkpoint(self, z, branchid):
        (u, d) = z.split()
        Z = z.function_space()
        mesh = z.function_space().mesh()
        comm = Z.mesh().mpi_comm()
        U = Function(Z, name = "U")
        U.assign(z)
        Filename = "XR/solution-%d.h5" % (branchid)
        with CheckpointFile(Filename, 'w', comm=comm) as afile:
            afile.save_mesh(mesh)  # optional
            afile.save_function(U)

    def monitor(self, params, branchid, solution, functionals):
        filename = "output/pvd/q-%s/jj-%s/solution-%d.pvd" % (params[0], params[4], branchid)
        pvd = File(filename, comm=solution.function_space().mesh().comm)
        self.save_pvd(solution, pvd, params)
        print("Wrote to %s" % filename)
        self.checkpoint(solution, branchid)

#        self.save_XR(solution, params, branchid)

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
    dc = DeflatedContinuation(problem=SmecticProblem(), teamsize=4, verbose=True, profile=False, logfiles=False)
    dc.run(values={"q": 320.27564, "W": 10, "d0":3, "ii": 8, "jj":8})
