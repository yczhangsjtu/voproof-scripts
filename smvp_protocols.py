from sympy import Symbol, latex, sympify, Integer, simplify
from sympy.abc import alpha, beta, gamma
from compiler.vo_protocol import VOProtocol
from compiler.symbol.vector import get_named_vector, PowerVector, UnitVector
from compiler.symbol.names import reset_name_counters, get_name
from compiler.symbol.matrix import Matrix
from compiler.symbol.util import rust_pk, rust_vk
from compiler.builder.latex import Math, AccumulationVector, ExpressionVector, \
    LaTeXBuilder, ProductAccumulationDivideVector, add_paren_if_not_atom, tex
from compiler.builder.rust import *


class SparseMVP(VOProtocol):
  def __init__(self):
    super(SparseMVP, self).__init__("SparseMVP")

  def preprocess(self, voexec, H, K, ell):
    n = voexec.vector_size
    u = get_named_vector("u")
    w = get_named_vector("w")
    v = get_named_vector("v")
    y = get_named_vector("y")
    voexec.preprocess_rust(rust_line_define_generator())
    voexec.preprocess_latex(Math(u).assign(
        ExpressionVector("\\gamma^{\\mathsf{row}_i}", ell))),
    voexec.preprocess_latex(Math(w).assign(
        ExpressionVector("\\gamma^{\\mathsf{col}_i}", ell))),
    voexec.preprocess_latex(Math(v).assign(
        ExpressionVector("\\mathsf{val}_i", ell))),
    voexec.preprocess_rust(
        rust_line_define_matrix_vectors(u, w, v, voexec.M, "gamma"))
    voexec.preprocess(Math(y).assign(u).circ(w),
                      rust_line_define_hadamard_vector(y, u, w))

    voexec.preprocess_vector(u, ell)
    voexec.preprocess_vector(w, ell)
    voexec.preprocess_vector(v, ell)
    voexec.preprocess_vector(y, ell)
    voexec.preprocess_output_pk(u)
    voexec.preprocess_output_pk(w)
    voexec.preprocess_output_pk(v)
    voexec.preprocess_output_pk(y)
    u._is_preprocessed = True
    voexec.u = u
    w._is_preprocessed = True
    voexec.w = w
    v._is_preprocessed = True
    voexec.v = v
    y._is_preprocessed = True
    voexec.y = y
    voexec.H = H
    voexec.K = K
    voexec.ell = ell
    return voexec

  def execute(self, voexec, a):
    n, H, K, ell = voexec.vector_size, voexec.H, voexec.K, voexec.ell
    M = voexec.M

    mu = Symbol(get_name("mu"))
    if voexec.rust_vector_size is None:
      voexec.rust_vector_size = voexec.verifier_redefine_symbol_rust(n, "n")
    rust_n = voexec.rust_vector_size
    rust_ell = voexec.verifier_redefine_symbol_rust(ell, "ell")
    voexec.verifier_rust_define_generator()
    voexec.verifier_send_randomness(mu)
    r = get_named_vector("r")
    voexec.prover_computes(
        Math(r).assign(ExpressionVector(
            "\\frac{1}{%s-\\gamma^i}" % tex(mu), H)),
        rust_line_define_expression_vector_inverse_i(
            r,
            rust_minus(mu, PowerVector(gamma, H).dumpr_at_index(sym_i)),
            H))
    c = get_named_vector("c")
    voexec.prover_computes(Math(c).assign()
                           .transpose(r, paren=False).append("\\boldsymbol{M}"),
                           rust_line_define_left_sparse_mvp_vector(
        c, rust_pk(M), r, H, K))
    s = get_named_vector("s")
    voexec.prover_computes(
        Math(s).assign(r).double_bar().paren(-c),
        rust_line_define_concat_neg_vector(s, r, c))

    voexec.prover_submit_vector(s, H + K)
    voexec.hadamard_query(
        mu * PowerVector(1, H) - PowerVector(gamma, H),
        s,
        PowerVector(1, H),
        PowerVector(1, H))
    voexec.inner_product_query(
        a.shift(n - H - K, rust_n - H - K),
        s.shift(n - H - K, rust_n - H - K))

    nu = Symbol(get_name("nu"))
    voexec.verifier_send_randomness(nu)

    h = get_named_vector("h")
    rnu = get_named_vector("rnu")
    voexec.prover_rust_define_expression_vector_inverse_i(
        rnu,
        rust_minus(nu, PowerVector(gamma, K).dumpr_at_index(sym_i)),
        K)
    voexec.prover_computes(Math(h).assign(
        ExpressionVector("\\frac{1}{%s-\\gamma^i}" % tex(nu), K)
    ).double_bar(
        ExpressionVector("\\frac{1}{(%s-%s)(%s-%s)}" %
                         (tex(mu), voexec.u.slice(Symbol("i")).dumps(),
                          tex(nu), voexec.w.slice(Symbol("i")).dumps()), ell)
    ),
        rust_line_define_concat_uwinverse_vector(
        h, rnu, mu, rust_pk(voexec.u), nu, rust_pk(voexec.w)
    ))
    voexec.prover_submit_vector(h, ell + K, rust_ell + K)

    voexec.hadamard_query(
        nu * PowerVector(1, K) - PowerVector(gamma, K),
        h,
        PowerVector(1, K),
        PowerVector(1, K),
    )

    voexec.hadamard_query(
        h,
        (mu * nu * PowerVector(1, ell, rust_ell) - mu * voexec.w -
         nu * voexec.u + voexec.y).shift(K),
        PowerVector(1, ell, rust_ell).shift(K),
        PowerVector(1, ell, rust_ell).shift(K),
    )

    voexec.inner_product_query(
        - h.shift(n - K, rust_n - K),
        s.shift(n - H - K, rust_n - H - K),
        h.shift(n - ell - K, rust_n - rust_ell - K),
        voexec.v.shift(n - ell, rust_n - rust_ell),
    )


class SparseMVPProverEfficient(VOProtocol):
  def __init__(self):
    super(SparseMVPProverEfficient, self).__init__("SparseMVPProverEfficient")

  def preprocess(self, voexec, H, K, ell):
    n = voexec.vector_size
    u = get_named_vector("u")
    w = get_named_vector("w")
    v = get_named_vector("v")
    y = get_named_vector("y")
    voexec.preprocess(Math(u).assign(
        ExpressionVector("\\gamma^{\\mathsf{row}_i}", ell)
    ), RustBuilder())
    voexec.preprocess(Math(w).assign(
        ExpressionVector("\\gamma^{\\mathsf{col}_i}", ell)
    ), RustBuilder())
    voexec.preprocess(Math(v).assign(
        ExpressionVector("\\mathsf{val}_i", ell)
    ), RustBuilder())
    voexec.preprocess(Math(y).assign(u).circ(w), RustBuilder())
    voexec.preprocess_vector(u, ell)
    voexec.preprocess_vector(w, ell)
    voexec.preprocess_vector(v, ell)
    voexec.preprocess_vector(y, ell)
    voexec.preprocess_output_pk(u)
    voexec.preprocess_output_pk(w)
    voexec.preprocess_output_pk(v)
    voexec.preprocess_output_pk(y)
    voexec.u = u
    voexec.w = w
    voexec.v = v
    voexec.y = y
    voexec.H = H
    voexec.K = K
    voexec.ell = ell
    return voexec

  def execute(self, voexec, a, b):
    n, H, K, ell = voexec.vector_size, voexec.H, voexec.K, voexec.ell

    mu = Symbol(get_name("mu"))
    voexec.verifier_send_randomness(mu)
    r = get_named_vector("r")
    voexec.prover_computes(Math(r).assign(
        ExpressionVector("\\frac{1}{\\alpha-\\gamma^i}", H)
    ), RustBuilder())
    voexec.prover_submit_vector(r, H)
    voexec.hadamard_query(
        mu * PowerVector(1, H) - PowerVector(gamma, H),
        r,
        PowerVector(1, H),
        PowerVector(1, H),
    )
    c = get_named_vector("c")
    voexec.prover_computes(Math(c).assign()
                           .transpose(r, paren=False).append("\\boldsymbol{M}"),
                           RustBuilder())
    voexec.prover_submit_vector(c, K)
    voexec.inner_product_query(
        a.shift(n - K),
        c.shift(n - K),
        b.shift(n - H),
        r.shift(n - H),
    )

    nu = Symbol(get_name("nu"))
    voexec.verifier_send_randomness(nu)

    r = get_named_vector("r")
    voexec.prover_computes(Math(r).assign(
        ExpressionVector("\\frac{1}{\\beta-\\gamma^i}", K)),
        RustBuilder())
    voexec.prover_submit_vector(r, K)

    t = get_named_vector("t")
    voexec.prover_computes(Math(t).assign(
        ExpressionVector("\\frac{1}{(\\alpha-%s)(\\beta-%s}" %
                         (voexec.u.slice(Symbol("i")).dumps(),
                          voexec.w.slice(Symbol("i")).dumps()), ell)),
        RustBuilder())
    voexec.prover_submit_vector(t, ell)

    voexec.hadamard_query(
        nu * PowerVector(1, K) - PowerVector(gamma, K),
        r,
        PowerVector(1, K),
        PowerVector(1, K),
    )

    voexec.hadamard_query(
        t,
        mu * nu * PowerVector(1, K) - mu * voexec.w -
        nu * voexec.u + voexec.y,
        PowerVector(1, K),
        PowerVector(1, K),
    )

    voexec.inner_product_query(
        t.shift(n - ell),
        voexec.v.shift(n - ell),
        c.shift(n - K),
        r.shift(n - K),
    )


class R1CS(VOProtocol):
  def __init__(self):
    super(R1CS, self).__init__("R1CS")

  def preprocess(self, voexec, H, K, sa, sb, sc):
    M = Matrix("M")
    voexec.pp_rust_init_size(H, "nrows")
    # voexec.pp_rust_init_size(K, "ncols")
    voexec.pp_rust_init_size(sa, "adensity")
    voexec.pp_rust_init_size(sb, "bdensity")
    voexec.pp_rust_init_size(sc, "cdensity")

    voexec.preprocess_rust(
        rust_line_concat_matrix_vertically(M, H,
                                           "cs.arows", "cs.brows", "cs.crows",
                                           "cs.acols", "cs.bcols", "cs.ccols",
                                           "cs.avals", "cs.bvals", "cs.cvals"))

    voexec.preprocess_output_pk(M)
    voexec.M = M
    M._is_preprocessed = True

    SparseMVP().preprocess(voexec, H * 3, K, sa + sb + sc)
    voexec.r1cs_H = H
    voexec.r1cs_K = K
    voexec.sa = sa
    voexec.sb = sb
    voexec.sc = sc
    return voexec

  def execute(self, voexec, x, w, ell):
    voexec.input_instance(x)
    voexec.input_witness(w)

    H, K, sa, sb, sc, n = voexec.r1cs_H, voexec.r1cs_K, \
        voexec.sa, voexec.sb, voexec.sc, voexec.vector_size
    M = voexec.M

    voexec.verifier_rust_define_vec(x, "x.instance.clone()")
    voexec.prover_rust_define_vec(w, "w.witness.clone()")
    voexec.verifier_rust_init_size(H, "nrows")
    voexec.verifier_rust_init_size(K, "ncols")
    voexec.verifier_rust_init_size(sa, "adensity")
    voexec.verifier_rust_init_size(sb, "bdensity")
    voexec.verifier_rust_init_size(sc, "cdensity")
    voexec.verifier_rust_init_size(ell, "input_size")
    voexec.try_verifier_redefine_vector_size_rust("n", n)
    rust_n = voexec.rust_vector_size

    u = get_named_vector("u")
    voexec.prover_computes(Math(u).assign().paren(
        LaTeXBuilder("\\boldsymbol{M}").paren(
            LaTeXBuilder(1).double_bar(x).double_bar(w)
        )
    ).double_bar(1).double_bar(x).double_bar(w),
        rust_line_define_sparse_mvp_concat_vector(
        u,
        rust_pk(M),
        rust_concat_and_one(x, w),
        H * 3, K
    ))

    voexec.prover_submit_vector(u, 3 * H + K)
    voexec.run_subprotocol(SparseMVP(), u)
    voexec.hadamard_query(
        u.shift(n-H, rust_n-H),
        u.shift(n-H*2, rust_n-H*2),
        PowerVector(1, H).shift(n-H, rust_n-H),
        u.shift(n-H*3, rust_n-H*3),
    )
    voexec.hadamard_query(
        PowerVector(1, ell+1).shift(H*3),
        u - x.shift(H*3+1) - UnitVector(H*3+1),
    )


class R1CSProverEfficient(VOProtocol):
  def __init__(self):
    super(R1CSProverEfficient, self).__init__("R1CS")

  def preprocess(self, voexec, H, K, s):
    SparseMVPProverEfficient().preprocess(voexec, H * 3, K, s)
    voexec.r1cs_H = H
    voexec.r1cs_K = K
    voexec.s = s
    return voexec

  def execute(self, voexec, x, w, ell):
    voexec.input_instance(x)
    voexec.input_witness(w)

    H, K, s, n = voexec.r1cs_H, voexec.r1cs_K, voexec.s, voexec.vector_size
    y = get_named_vector("y")
    voexec.prover_computes(Math(y).assign().paren(
        Math("\\boldsymbol{M}").paren(Math(1).double_bar(x).double_bar(w))
    ), RustBuilder())
    voexec.prover_submit_vector(y, 3 * H)
    voexec.prover_submit_vector(w, H - ell - 1)
    voexec.run_subprotocol(SparseMVPProverEfficient(),
                           UnitVector(1) + x.shift(1) +
                           w.shift(ell + 1), y)
    voexec.hadamard_query(
        y.shift(n-H),
        y.shift(n-H*2),
        PowerVector(1, H).shift(n-H),
        y.shift(n-H*3),
    )


class HPR(VOProtocol):
  def __init__(self):
    super(HPR, self).__init__("HPR")

  def preprocess(self, voexec, H, K, sa, sb, sc, sd):
    M = Matrix("M")
    d = get_named_vector("d")
    # voexec.preprocess_rust(rust_line_init_size(H, "nrows"))
    voexec.preprocess_rust(rust_line_init_size(K, "ncols"))
    voexec.preprocess_rust(rust_line_init_size(sa, "adensity"))
    voexec.preprocess_rust(rust_line_init_size(sb, "bdensity"))
    voexec.preprocess_rust(rust_line_init_size(sc, "cdensity"))
    voexec.preprocess_rust(rust_line_init_size(sd, "ddensity"))

    voexec.preprocess_rust(
        rust_line_concat_matrix_horizontally(
            M, K,
            "cs.arows", "cs.brows", "cs.crows",
            "cs.acols", "cs.bcols", "cs.ccols",
            "cs.avals", "cs.bvals", "cs.cvals",
            "cs.drows", "cs.dvals"))

    voexec.preprocess_output_pk(M)
    voexec.M = M
    M._is_preprocessed = True

    SparseMVP().preprocess(voexec, H, K * 3 + 1, sa + sb + sc + sd)
    voexec.hpr_H = H
    voexec.hpr_K = K
    voexec.sa = sa
    voexec.sb = sb
    voexec.sc = sc
    voexec.sd = sd
    voexec.d = d

    return voexec

  def execute(self, voexec, x, w1, w2, w3):
    voexec.input_instance(x)
    voexec.input_witness(w1)
    voexec.input_witness(w2)
    voexec.input_witness(w3)

    H, K, sa, sb, sc, sd, n = voexec.hpr_H, voexec.hpr_K, \
        voexec.sa, voexec.sb, voexec.sc, voexec.sd, voexec.vector_size

    voexec.verifier_rust_define_vec(x, "x.instance.clone()")
    voexec.prover_rust_define_vec(w1, "w.witness.0.clone()")
    voexec.prover_rust_define_vec(w2, "w.witness.1.clone()")
    voexec.prover_rust_define_vec(w3, "w.witness.2.clone()")

    voexec.verifier_rust_init_size(H, "nrows")
    voexec.verifier_rust_init_size(K, "ncols")
    voexec.verifier_rust_init_size(sa, "adensity")
    voexec.verifier_rust_init_size(sb, "bdensity")
    voexec.verifier_rust_init_size(sc, "cdensity")
    voexec.verifier_rust_init_size(sd, "ddensity")

    voexec.try_verifier_redefine_vector_size_rust("n", n)
    rust_n = voexec.rust_vector_size

    w = get_named_vector("w")
    voexec.prover_computes_latex(Math(w).assign(
        w1).double_bar(w2).double_bar(w3))
    voexec.prover_rust_define_concat_vector(w, w1, w2, w3)
    voexec.prover_submit_vector(w, 3 * K)
    voexec.run_subprotocol(SparseMVP(),
                           x + w.shift(H + 1) +
                           UnitVector(H + 1))
    voexec.hadamard_query(
        w.shift(n-K, rust_n-K),
        w.shift(n-K*2, rust_n-K*2),
        PowerVector(1, K).shift(n-K, rust_n-K),
        w.shift(n-K*3, rust_n-K*3))


class HPRProverEfficient(VOProtocol):
  def __init__(self):
    super(HPRProverEfficient, self).__init__("HPR")

  def preprocess(self, voexec, H, K, s):
    SparseMVPProverEfficient().preprocess(voexec, H, K * 3 + 1, s)
    voexec.hpr_H = H
    voexec.hpr_K = K
    voexec.s = s
    return voexec

  def execute(self, voexec, x, w1, w2, w3, ell):
    voexec.input_instance(x)
    voexec.input_witness(w1)
    voexec.input_witness(w2)
    voexec.input_witness(w3)

    H, K, s, n = voexec.hpr_H, voexec.hpr_K, voexec.s, voexec.vector_size
    w = get_named_vector("w")
    voexec.prover_computes(Math(w).assign(
        w1).double_bar(w2).double_bar(w3), RustBuilder())
    voexec.prover_submit_vector(w, 3 * K)
    voexec.run_subprotocol(SparseMVPProverEfficient(),
                           w.shift(1) + UnitVector(1), x)
    voexec.hadamard_query(
        w.shift(n-K),
        w.shift(n-K*2),
        PowerVector(1, K).shift(n-K),
        w.shift(n-K*3),
    )
