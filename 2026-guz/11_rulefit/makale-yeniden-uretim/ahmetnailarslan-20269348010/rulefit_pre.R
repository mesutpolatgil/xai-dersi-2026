# Kullanım: Rscript rulefit_pre.R <klasor> [ikisi|ham] [yalniz_kat]
# Klasörde ham.csv, kutu.csv (son sütun hedef) ve kat.csv bulunur; kurallar.csv ve tahmin.csv yazılır.
# pre varsayılanları: ntrees=500, maxdepth=3, learnrate=0.01, sampfrac=0.5, winsfrac=0.025, nfolds=10;
# kural ağırlıkları Lasso (glmnet), lambda = lambda.1se (coef.pre ve predict.pre varsayılanı).
arg <- commandArgs(TRUE)
d <- arg[1]
mod <- if (length(arg) > 1) arg[2] else "ikisi"
lib <- Sys.getenv("R_LIBS_USER"); if (dir.exists(lib)) .libPaths(c(lib, .libPaths()))
suppressMessages(library(pre))
ham <- read.csv(file.path(d, "ham.csv"))
kutu <- if (mod == "ikisi") read.csv(file.path(d, "kutu.csv"))
kat <- read.csv(file.path(d, "kat.csv"))$kat
set.seed(as.integer(read.csv(file.path(d, "kat.csv"))$tohum[1]))
# 0-1 kural bulunursa glmnet "2 or more columns" hatası verir: kuralsız (sabit) model say.
uydur <- function(v) tryCatch(suppressWarnings(pre(hedef ~ ., data = v, type = "rules")),
  error = function(e) if (grepl("2 or more columns", conditionMessage(e))) NULL else stop(e))
say <- function(m) {
  if (is.null(m)) return(data.frame())
  cf <- coef(m)
  cf[cf$coefficient != 0 & cf$rule != "(Intercept)", ]
}
tahmin <- rep(NA_real_, nrow(ham)); cikti <- list()
ekle <- function(k, tur, cf) if (nrow(cf) > 0)
  cikti[[length(cikti) + 1]] <<- data.frame(kat = k, tur = tur, kural = cf$description)
katlar <- if (length(arg) > 2) as.integer(arg[3]) else sort(unique(kat))
for (k in katlar) {
  m <- uydur(ham[kat != k, ])
  tahmin[kat == k] <- if (is.null(m)) mean(ham$hedef[kat != k]) else predict(m, newdata = ham[kat == k, ])
  ekle(k, "ham", say(m))
  if (mod == "ikisi") ekle(k, "kutu", say(uydur(kutu[kat != k, ])))
}
bos <- data.frame(kat = integer(0), tur = character(0), kural = character(0))
write.csv(do.call(rbind, c(list(bos), cikti)), file.path(d, "kurallar.csv"), row.names = FALSE)
write.csv(data.frame(tahmin = tahmin), file.path(d, "tahmin.csv"), row.names = FALSE)
