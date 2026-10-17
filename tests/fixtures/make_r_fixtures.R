# Reference outputs from the R package FSM, used by the Python parity tests.
#
# Calls only FSM's exported API and records inputs and outputs as JSON, so the
# Python suite can check parity without R. Regenerate from the repo root with
#   Rscript tests/fixtures/make_r_fixtures.R
# and commit the result; the r-parity workflow fails if a regeneration differs.

suppressMessages({
  library(FSM)
  library(jsonlite)
})

out_dir <- file.path("tests", "fixtures", "r")
dir.create(out_dir, showWarnings = FALSE, recursive = TRUE)
unlink(list.files(out_dir, full.names = TRUE))

write_case <- function(name, obj) {
  writeLines(
    toJSON(obj, digits = NA, auto_unbox = TRUE, null = "null", na = "null",
           matrix = "rowmajor"),
    file.path(out_dir, paste0(name, ".json"))
  )
}

# Runs `expr`, returning list(value=...) or list(error=message) so that R's
# failure modes are recorded as part of the reference behavior.
capture <- function(expr) {
  tryCatch(list(value = expr), error = function(e) list(error = conditionMessage(e)))
}

# ---------------------------------------------------------------- datasets
data(Lalonde)
datasets <- list()
datasets$lalonde <- Lalonde
set.seed(101)
datasets$lalonde60 <- {
  d <- Lalonde[sort(sample(nrow(Lalonde), 60)), ]
  d$Index <- seq_len(nrow(d))
  d
}
set.seed(102)
datasets$gauss <- data.frame(Index = 1:40, matrix(rnorm(40 * 4), 40))
set.seed(103)
datasets$heavy <- data.frame(Index = 1:36, matrix(rt(36 * 3, df = 3), 36))
set.seed(104)
datasets$binary <- data.frame(Index = 1:30, matrix(rbinom(30 * 4, 1, 0.4), 30))
set.seed(105)
datasets$collinear <- {
  x1 <- rnorm(30)
  x2 <- rnorm(30)
  data.frame(Index = 1:30, x1 = x1, x2 = x2, x3 = x1 + x2)
}
# Table 1 of Chattopadhyay, Morris and Zubizarreta (2021): 12 Lalonde ages.
datasets$ages12 <- data.frame(
  Index = 1:12, Age = c(19, 20, 20, 20, 23, 24, 24, 25, 25, 28, 31, 41)
)
for (nm in names(datasets)) {
  d <- datasets[[nm]]
  write_case(paste0("data_", nm), list(columns = names(d), rows = as.matrix(d)))
}

# ---------------------------------------------------------------- som
som_cases <- list()
set.seed(201)
for (cfg in list(
  list(id = "scomars_eq", n_treat = 2, sizes = c(20, 20), method = "SCOMARS"),
  list(id = "scomars_uneq", n_treat = 2, sizes = c(12, 24), method = "SCOMARS"),
  list(id = "scomars_ctrl", n_treat = 2, sizes = c(15, 15), method = "SCOMARS",
       control = TRUE),
  list(id = "gp_k3", n_treat = 3, sizes = c(10, 12, 14), method = "global percentage"),
  list(id = "gp_k4", n_treat = 4, sizes = c(5, 5, 5, 5), method = "global percentage"),
  list(id = "chunk_k3", n_treat = 3, sizes = c(8, 8, 8), method = "randomized chunk"),
  list(id = "chunk_uneq", n_treat = 2, sizes = c(6, 12), method = "randomized chunk"),
  list(id = "chunk_discard", n_treat = 2, sizes = c(5, 5), method = "randomized chunk",
       discard_n = 14)
)) {
  control <- isTRUE(cfg$control)
  n <- sum(cfg$sizes)
  marg <- if (cfg$method == "SCOMARS") rep(cfg$sizes[2] / n, n) else NULL
  df <- if (!is.null(cfg$discard_n)) data.frame(a = seq_len(cfg$discard_n)) else NULL
  draw <- function() {
    som(data_frame = df, n_treat = cfg$n_treat, treat_sizes = cfg$sizes,
        include_discard = !is.null(df), method = cfg$method, control = control,
        marginal_treat = marg)
  }
  one <- draw()
  reps <- 10000
  draws <- t(replicate(reps, draw()$Treat))
  levels <- sort(unique(as.vector(draws)))
  stage_freq <- sapply(levels, function(l) colMeans(draws == l))
  write_case(paste0("som_", cfg$id), list(
    config = cfg, control = control, marginal_treat = marg,
    one = as.list(one), reps = reps, levels = levels,
    stage_freq = stage_freq
  ))
}

# ---------------------------------------------------------------- fsm
s_functions <- c("constant", "Dopt", "Aopt", "negative Dopt", "max pc", "min pc",
                 "Dopt pc", "max average", "min average", "Dopt average",
                 "marginal var sum")

make_som <- function(n, k, seed) {
  set.seed(seed)
  if (k == 2) {
    n2 <- n %/% 2
    som(n_treat = 2, treat_sizes = c(n - n2, n2), method = "SCOMARS",
        marginal_treat = rep(n2 / n, n))
  } else {
    som(n_treat = k, treat_sizes = rep(n / k, k), method = "randomized chunk")
  }
}

run_fsm <- function(id, dname, k, seed, ...) {
  d <- datasets[[dname]]
  s <- make_som(nrow(d), k, seed)
  args <- list(...)
  res <- capture(do.call(fsm, c(list(data_frame = d, SOM = s, ties = "smallest",
                                     units_print = FALSE), args)))
  rec <- list(dataset = dname, som = s$Treat, args = args)
  if (!is.null(res$error)) {
    rec$error <- res$error
  } else {
    f <- res$value
    rec$z <- f$data_frame_allocated$Treat
    rec$selected <- f$som_appended$Index
    rec$criteria <- f$criteria
  }
  write_case(paste0("fsm_", id), rec)
}

aopt_args <- function(k_cov) {
  pol <- matrix(0, 2, k_cov + 1)
  pol[1, 2] <- 1
  pol[2, 3] <- 1
  list(Pol_mat = pol, w_pol = c(0.7, 0.3))
}

for (dname in c("lalonde60", "gauss", "heavy", "binary", "ages12")) {
  k_cov <- ncol(datasets[[dname]]) - 1
  for (sf in s_functions) {
    if (k_cov < 2 && sf == "Aopt") next
    extra <- if (sf == "Aopt") aopt_args(k_cov) else list()
    id <- paste(dname, gsub(" ", "_", sf), sep = "_")
    do.call(run_fsm, c(list(id = id, dname = dname, k = 2, seed = 301), list(s_function = sf),
                       extra))
  }
}
# Options grid on the D-optimal criterion.
run_fsm("gauss_k4", "gauss", 4, 302, s_function = "Dopt")
run_fsm("lalonde60_k3", "lalonde60", 3, 303, s_function = "Dopt")
run_fsm("heavy_k3", "heavy", 3, 304, s_function = "Dopt")
run_fsm("gauss_noint", "gauss", 2, 305, s_function = "Dopt", intercept = FALSE)
run_fsm("gauss_nostd", "gauss", 2, 306, s_function = "Dopt", standardize = FALSE)
run_fsm("lalonde60_nostd", "lalonde60", 2, 307, s_function = "Dopt", standardize = FALSE)
run_fsm("gauss_eps1", "gauss", 2, 308, s_function = "Dopt", eps = 0.1)
run_fsm("gauss_eps5", "gauss", 2, 309, s_function = "Dopt", eps = 1e-5)
run_fsm("gauss_qinit", "gauss", 2, 310, s_function = "Dopt",
        Q_initial = diag(5) * 2)
run_fsm("collinear_dopt", "collinear", 2, 311, s_function = "Dopt")
run_fsm("collinear_qinit", "collinear", 2, 312, s_function = "Dopt",
        Q_initial = diag(4))
run_fsm("binary_k3", "binary", 3, 313, s_function = "Dopt")
run_fsm("ages12_paper", "ages12", 2, 314, s_function = "Dopt")

# Full Lalonde, as in Section 3 of the R paper (N = 445, n1 = 222).
set.seed(315)
s_full <- som(n_treat = 2, treat_sizes = c(222, 223), method = "SCOMARS",
              marginal_treat = rep(223 / 445, 445))
f_full <- fsm(Lalonde, SOM = s_full, ties = "smallest", units_print = FALSE)
write_case("fsm_lalonde_full", list(
  dataset = "lalonde", som = s_full$Treat, args = list(s_function = "Dopt"),
  z = f_full$data_frame_allocated$Treat, selected = f_full$som_appended$Index
))

# ties = "random": per-unit group frequencies over repeated runs, same SOM.
d <- datasets$binary
s <- make_som(nrow(d), 2, 316)
reps <- 2000
zz <- t(replicate(reps, fsm(d, SOM = s, ties = "random", units_print = FALSE)
                  $data_frame_allocated$Treat))
write_case("fsm_ties_random_binary", list(
  dataset = "binary", som = s$Treat, reps = reps, freq_group1 = colMeans(zz == 1)
))

# ---------------------------------------------------------------- fsm_batch
for (cfg in list(
  list(id = "gauss", dname = "gauss", k = 2, split = 20, seed = 401),
  list(id = "lalonde60_k3", dname = "lalonde60", k = 3, split = 30, seed = 402),
  list(id = "gauss_noint", dname = "gauss", k = 2, split = 16, seed = 403,
       intercept = FALSE)
)) {
  d <- datasets[[cfg$dname]]
  first <- d[seq_len(cfg$split), ]
  second <- d[(cfg$split + 1):nrow(d), ]
  intercept <- if (is.null(cfg$intercept)) TRUE else cfg$intercept
  s1 <- make_som(nrow(first), cfg$k, cfg$seed)
  f1 <- fsm(first, SOM = s1, ties = "smallest", units_print = FALSE,
            intercept = intercept)
  s2 <- make_som(nrow(second), cfg$k, cfg$seed + 50)
  fb <- fsm_batch(data_frame = second, data_frame_past = f1$data_frame_allocated,
                  t_ind = "Treat", SOM = s2, ties = "smallest", units_print = FALSE,
                  intercept = intercept)
  write_case(paste0("batch_", cfg$id), list(
    dataset = cfg$dname, split = cfg$split, intercept = intercept,
    past_z = f1$data_frame_allocated$Treat, som = s2$Treat,
    z = fb$data_frame_allocated$Treat, selected = fb$som_appended$Index,
    criteria = fb$criteria
  ))
}

# ---------------------------------------------------------------- crd
set.seed(501)
for (cfg in list(list(id = "k2", sizes = c(6, 6), control = FALSE),
                 list(id = "k3_ctrl", sizes = c(4, 6, 8), control = TRUE))) {
  df <- data.frame(Index = seq_len(sum(cfg$sizes)))
  reps <- 10000
  zz <- t(replicate(reps, crd(df, n_treat = length(cfg$sizes), treat_sizes = cfg$sizes,
                               control = cfg$control)$Treat))
  levels <- sort(unique(as.vector(zz)))
  write_case(paste0("crd_", cfg$id), list(
    sizes = cfg$sizes, control = cfg$control, reps = reps, levels = levels,
    one = zz[1, ], unit_freq = sapply(levels, function(l) colMeans(zz == l))
  ))
}

# ---------------------------------------------------------------- inference
set.seed(601)
n <- 24
x <- rnorm(n)
y1 <- 10 + 2 * x + rnorm(n)
y2 <- y1 + 1.5
y3 <- y1 - 0.5
alloc2 <- t(replicate(300, sample(rep(1:2, n / 2))))
alloc3 <- t(replicate(300, sample(rep(1:3, n / 3))))
yobs2 <- ifelse(alloc2[1, ] == 1, y1, y2)
yobs3 <- cbind(y1, y2, y3)[cbind(seq_len(n), alloc3[1, ])]
yobs_ties <- round(yobs2)
perm <- list()
for (method in c("marginal mean", "marginal rank")) {
  for (alt in c("not equal", "greater", "less")) {
    for (cc in list(list(id = "k2", y = yobs2, a = alloc2, contrast = c(1, -1), tau = 0),
                    list(id = "k2tau", y = yobs2, a = alloc2, contrast = c(1, -1), tau = -1),
                    list(id = "k2rev", y = yobs2, a = alloc2, contrast = c(-1, 1), tau = 0),
                    list(id = "k2ties", y = yobs_ties, a = alloc2, contrast = c(1, -1),
                         tau = 0),
                    list(id = "k3", y = yobs3, a = alloc3, contrast = c(1, -1, 0),
                         tau = 0))) {
      res <- capture(suppressWarnings(perm_test(
        Y_obs = cc$y, alloc_obs = cc$a[1, ], alloc = cc$a, contrast = cc$contrast,
        tau = cc$tau, method = method, alternative = alt)))
      perm[[length(perm) + 1]] <- list(
        id = cc$id, method = method, alternative = alt, contrast = cc$contrast,
        tau = cc$tau, y_obs = cc$y, alloc = cc$a, result = res
      )
    }
  }
}
write_case("perm_test", perm)

# ess_rand: three designs, 200 iterations each.
set.seed(602)
xa <- rnorm(n)
ymat <- cbind(5 + xa + rnorm(n), 5 + xa + rnorm(n) + 1)
d_crd <- t(replicate(200, sample(rep(1:2, n / 2))))
d_blk <- t(replicate(200, {
  z <- integer(n)
  o <- order(xa)
  for (b in seq(1, n, 2)) z[o[b:(b + 1)]] <- sample(1:2)
  z
}))
d_fsm <- t(replicate(200, {
  s <- som(n_treat = 2, treat_sizes = c(n / 2, n / 2), method = "SCOMARS",
           marginal_treat = rep(0.5, n))
  fsm(data.frame(Index = 1:n, x = xa), SOM = s, units_print = FALSE)$data_frame_allocated$Treat
}))
arr <- array(0, dim = c(200, n, 3))
arr[, , 1] <- d_crd
arr[, , 2] <- d_blk
arr[, , 3] <- d_fsm
write_case("ess_rand", list(
  assign = lapply(1:3, function(j) arr[, , j]), y_mat = ymat,
  contrast = c(1, -1),
  result = ess_rand(assign_array = arr, Y_mat = ymat, contrast = c(1, -1)),
  result_rev = ess_rand(assign_array = arr, Y_mat = ymat, contrast = c(-1, 1))
))

# ess_model: X with two covariates, five assignment vectors (columns).
xm <- cbind(xa, xa^2)
amat <- cbind(d_crd[1, ], d_crd[2, ], d_blk[1, ], d_fsm[1, ], d_fsm[2, ])
write_case("ess_model", list(
  x_cov = xm, assign = amat, y_mat = ymat, contrast = c(1, -1),
  result = ess_model(X_cov = xm, assign_matrix = amat, Y_mat = ymat, contrast = c(1, -1))
))
# Unequal group sizes (10 vs 14) separate the candidate residual-variance
# denominators, which coincide when groups are the same size.
set.seed(604)
amat_uneq <- sapply(1:5, function(i) sample(rep(1:2, c(10, 14))))
write_case("ess_model_unequal", list(
  x_cov = xm, assign = amat_uneq, y_mat = ymat, contrast = c(1, -1),
  result = ess_model(X_cov = xm, assign_matrix = amat_uneq, Y_mat = ymat,
                     contrast = c(1, -1))
))

# tasmd_rand: vector and matrix inputs, both denominators, custom targets.
d <- datasets$lalonde60
set.seed(603)
a1 <- t(replicate(50, sample(rep(1:2, 30))))
a2 <- t(replicate(50, {
  s <- som(n_treat = 2, treat_sizes = c(30, 30), method = "SCOMARS",
           marginal_treat = rep(0.5, 60))
  fsm(d, SOM = s, units_print = FALSE)$data_frame_allocated$Treat
}))
tas <- list()
add_tasmd <- function(id, ...) {
  args <- list(...)
  res <- capture(do.call(tasmd_rand, c(list(data_frame = d, index_col = TRUE), args)))
  if (!is.null(res$value) && is.list(res$value) && !is.data.frame(res$value)) {
    res$value <- lapply(res$value, function(v) {
      if (is.matrix(v)) list(rownames = rownames(v), colnames = colnames(v), values = v) else v
    })
  } else if (!is.null(res$value)) {
    res$value <- list(rownames = rownames(res$value), colnames = colnames(res$value),
                      values = as.matrix(res$value))
  }
  tas[[length(tas) + 1]] <<- list(id = id, args = args, result = res)
}
add_tasmd("vec_target", alloc1 = a1[1, ], alloc2 = a2[1, ], roundoff = 10)
add_tasmd("vec_group", alloc1 = a1[1, ], alloc2 = a2[1, ], denom = "group", roundoff = 10)
add_tasmd("vec_lab2", alloc1 = a1[1, ], alloc2 = a2[1, ], treat_lab = 2, roundoff = 10)
add_tasmd("vec_round3", alloc1 = a1[1, ], alloc2 = a2[1, ])
add_tasmd("vec_custom", alloc1 = a1[1, ], alloc2 = a2[1, ],
          mean_tar = colMeans(d[, -1]) + 0.1, sd_tar = apply(d[, -1], 2, sd) * 1.5,
          roundoff = 10)
add_tasmd("mat_target", alloc1 = a1, alloc2 = a2, roundoff = 10)
add_tasmd("mat_round3", alloc1 = a1, alloc2 = a2)
add_tasmd("mat_mismatch", alloc1 = a1, alloc2 = a2[1:10, ], roundoff = 10)
write_case("tasmd_rand", list(data = "lalonde60", cases = tas))

# ---------------------------------------------------------------- make_sq_inter
msi <- list()
base <- data.frame(male = c(rep(1, 6), rep(0, 6)),
                   age = c(20, 30, 40, 40, 50, 60, 20, 30, 40, 40, 50, 60),
                   edu = c(10, 12, 8, 16, 12, 14, 9, 11, 12, 16, 18, 10))
for (sq in c(TRUE, FALSE)) for (it in c(TRUE, FALSE)) for (km in c(TRUE, FALSE)) {
  res <- capture(make_sq_inter(base, is_square = sq, is_inter = it, keep_marginal = km))
  if (!is.null(res$value)) {
    res$value <- list(columns = names(res$value), values = as.matrix(res$value))
  }
  msi[[length(msi) + 1]] <- list(is_square = sq, is_inter = it, keep_marginal = km,
                                 result = res)
}
write_case("make_sq_inter", list(data = list(columns = names(base), values = as.matrix(base)),
                                 cases = msi))

# ---------------------------------------------------------------- love_plot
png(file.path("tests", "fixtures", "love_plot_r.png"), width = 900, height = 650, res = 120)
love_plot(d, index_col = TRUE, alloc1 = a1[1, ], alloc2 = a2[1, ], imbalance = "TASMD",
          legend_text = c("CRD", "FSM"))
invisible(dev.off())

write_case("manifest", list(
  r_version = R.version.string,
  fsm_version = as.character(packageVersion("FSM")),
  jsonlite_version = as.character(packageVersion("jsonlite"))
))
