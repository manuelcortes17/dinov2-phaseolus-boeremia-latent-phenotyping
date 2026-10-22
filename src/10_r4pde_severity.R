# 10_r4pde_severity.R
# Palette-based leaf disease severity (R4PDE color-palette workflow, in R, no pliman).
# Classes (palettes): healthy (green), lesion (chlorotic+necrotic), background (gray),
# dark (inoculation plug / card-black), light (card-white / specular). Severity =
# lesion / (healthy + lesion) * 100 (% diseased leaf area). Epidemiology (AUDPC) follows
# R4PDE conventions (Del Ponte, r4pde.net).
suppressWarnings(suppressMessages({library(jpeg)}))
for(loc in c("C.UTF-8","en_US.UTF-8","C.utf8")) if(!inherits(try(Sys.setlocale("LC_ALL",loc),silent=TRUE),"try-error")) break
set.seed(1)

# ---- palette reference colors (sRGB 0-1); multiple samples per class ----
PAL <- list(
  healthy = rbind(c(.25,.45,.15),c(.35,.55,.20),c(.20,.40,.12),c(.45,.60,.25),
                  c(.30,.50,.18),c(.15,.35,.10),c(.40,.58,.22)),
  lesion  = rbind(c(.78,.78,.22),c(.68,.62,.16),c(.82,.80,.38),c(.72,.68,.20),  # chlorotic
                  c(.50,.40,.20),c(.40,.30,.15),c(.60,.48,.28),c(.34,.25,.13),  # necrotic
                  c(.55,.45,.18),c(.66,.55,.30),
                  c(.28,.20,.12),c(.22,.16,.10),c(.32,.24,.14),c(.24,.18,.11)), # dark-brown necrosis
  backg   = rbind(c(.80,.82,.82),c(.70,.72,.72),c(.88,.90,.90),c(.60,.62,.63),
                  c(.75,.78,.79),c(.66,.69,.70)),
  dark    = rbind(c(.05,.05,.05),c(.14,.14,.14),c(.02,.02,.02),c(.20,.20,.21)),
  light   = rbind(c(.96,.96,.96),c(1,1,1),c(.90,.90,.92),c(.98,.97,.95))
)
CLASSES <- names(PAL)
# convert palettes to Lab
to_lab <- function(rgb) convertColor(rgb, from="sRGB", to="Lab")
PAL_LAB <- lapply(PAL, to_lab)

classify <- function(img){  # img: HxWx3 array 0-1 -> matrix of class idx
  h <- dim(img)[1]; w <- dim(img)[2]
  px <- matrix(img, ncol=3)                      # (h*w,3) column-major
  lab <- convertColor(px, from="sRGB", to="Lab") # (n,3)
  n <- nrow(lab)
  best <- rep(1L, n); bestd <- rep(Inf, n)
  for(ci in seq_along(CLASSES)){
    P <- PAL_LAB[[ci]]
    # min dist to any sample in this class
    dmin <- rep(Inf, n)
    for(j in seq_len(nrow(P))){
      d <- (lab[,1]-P[j,1])^2 + (lab[,2]-P[j,2])^2 + (lab[,3]-P[j,3])^2
      dmin <- pmin(dmin, d)
    }
    upd <- dmin < bestd; best[upd] <- ci; bestd[upd] <- dmin[upd]
  }
  matrix(best, nrow=h)
}

severity_of <- function(path, step=1L){
  img <- readJPEG(path)
  if(length(dim(img))==2) img <- array(rep(img,3), dim=c(dim(img),3))
  if(step>1L){ h<-dim(img)[1]; w<-dim(img)[2]
    img <- img[seq(1,h,by=step), seq(1,w,by=step), , drop=FALSE] }
  cls <- classify(img)
  hn <- sum(cls==which(CLASSES=="healthy"))
  ln <- sum(cls==which(CLASSES=="lesion"))
  leaf <- hn + ln
  sev <- if(leaf>0) 100*ln/leaf else NA_real_
  list(sev=sev, healthy_px=hn, lesion_px=ln, leaf_px=leaf, cls=cls, img=img)
}

# ---- validation montage on representative images ----
args <- commandArgs(trailingOnly=TRUE)
mode <- ifelse(length(args)>=1, args[1], "validate")
idx <- read.csv("/home/claude/proj/outputs/index/master_index.csv",
                stringsAsFactors=FALSE, fileEncoding="UTF-8")
idx <- idx[idx$exists=="True" | idx$exists==TRUE, ]

if(mode=="validate"){
  pick <- function(acc,tr,tp){
    s <- idx[idx$accession==acc & idx$treatment==tr & idx$timepoint==tp & idx$is_augmented=="False",]
    if(nrow(s)==0) return(NULL); s$img_path[1]
  }
  sel <- list(
    c("ASC 160","control","T1"), c("ASC 160","inoculated","T1"),
    c("ASC 160","inoculated","T2"), c("ASC 160","inoculated","T3"),
    c("CAL 96","inoculated","T3"), c("CAL 96","control","T3"))
  png("/tmp/sev_validate.png", width=1500, height=1000, res=110)
  par(mfrow=c(length(sel),3), mar=c(1,1,2,1))
  palcol <- c(healthy="#2ca02c", lesion="#d62728", backg="#dddddd", dark="#000000", light="#ffffff")
  for(s in sel){
    p <- pick(s[1],s[2],s[3]); if(is.null(p)) next
    r <- severity_of(p)
    plot(as.raster(r$img)); title(sprintf("%s %s %s", s[1],s[2],s[3]), cex.main=.9)
    cm <- matrix(palcol[CLASSES[r$cls]], nrow=nrow(r$cls))
    plot(as.raster(cm)); title("palette classes", cex.main=.9)
    les <- r$img; m <- r$cls==which(CLASSES=="lesion")
    R<-les[,,1];G<-les[,,2];B<-les[,,3]; R[m]<-1;G[m]<-0;B[m]<-0
    les[,,1]<-R;les[,,2]<-G;les[,,3]<-B
    plot(as.raster(les)); title(sprintf("lesion  sev=%.1f%%", r$sev), cex.main=.9)
    cat(sprintf("%-8s %-10s %s  sev=%.2f  leaf_px=%d\n", s[1],s[2],s[3], r$sev, r$leaf_px))
  }
  dev.off()
  cat("wrote /tmp/sev_validate.png\n")
}

if(mode=="all"){
  outdir <- "/home/claude/proj/outputs/r4pde"; dir.create(outdir, showWarnings=FALSE, recursive=TRUE)
  logf <- "/home/claude/proj/logs/severity.log"
  o <- idx[idx$is_augmented=="False" & idx$is_copy!="True" & idx$timepoint %in% c("T1","T2","T3"), ]
  n <- nrow(o); cat("severity for", n, "original leaves\n")
  res <- data.frame(image_id=character(n), key=character(n), accession=character(n),
    unit=character(n), treatment=character(n), timepoint=character(n),
    biological_unit_id=character(n), severity=NA_real_, lesion_px=NA_integer_,
    healthy_px=NA_integer_, leaf_px=NA_integer_, stringsAsFactors=FALSE)
  t0 <- Sys.time()
  for(i in seq_len(n)){
    r <- tryCatch(severity_of(o$img_path[i], step=2L), error=function(e) NULL)
    res$image_id[i]<-o$image_id[i]; res$key[i]<-o$key[i]; res$accession[i]<-o$accession[i]
    res$unit[i]<-o$unit[i]; res$treatment[i]<-o$treatment[i]; res$timepoint[i]<-o$timepoint[i]
    res$biological_unit_id[i]<-o$biological_unit_id[i]
    if(!is.null(r)){ res$severity[i]<-r$sev; res$lesion_px[i]<-r$lesion_px
      res$healthy_px[i]<-r$healthy_px; res$leaf_px[i]<-r$leaf_px }
    if(i %% 100 == 0){ el<-as.numeric(difftime(Sys.time(),t0,units="secs"))
      cat(sprintf("%d/%d %.0fs %.2f img/s\n", i, n, el, i/el), file=logf, append=TRUE) }
  }
  write.csv(res, file.path(outdir,"severity_per_image.csv"), row.names=FALSE)
  # ---- per unit x timepoint (mean over technical images) ----
  agg <- aggregate(severity ~ biological_unit_id + accession + treatment + timepoint,
                   data=res, FUN=function(x) mean(x, na.rm=TRUE))
  write.csv(agg, file.path(outdir,"severity_per_unit_time.csv"), row.names=FALSE)
  # ---- AUDPC across T1,T2,T3 (R4PDE convention; equally spaced, dt=1) ----
  tnum <- c(T1=1,T2=2,T3=3)
  wide <- reshape(agg[,c("biological_unit_id","accession","treatment","timepoint","severity")],
                  idvar=c("biological_unit_id","accession","treatment"),
                  timevar="timepoint", direction="wide")
  audpc_tp <- function(y, t=c(1,2,3)){ # trapezoidal
    ok <- !is.na(y); if(sum(ok)<2) return(NA_real_)
    y<-y[ok]; t<-t[ok]; sum(diff(t)*(head(y,-1)+tail(y,-1))/2) }
  sv <- as.matrix(wide[,grep("^severity\\.", names(wide))])
  cn <- gsub("severity\\.","",colnames(sv)); ord <- match(c("T1","T2","T3"), cn)
  sv <- sv[, ord, drop=FALSE]
  wide$AUDPC <- apply(sv, 1, audpc_tp)
  wide$rAUDPC <- wide$AUDPC / (100 * (3-1))   # relative to max possible (severity 100 over 2 intervals)
  wide$sev_final_T3 <- sv[,3]
  write.csv(wide, file.path(outdir,"audpc_per_unit.csv"), row.names=FALSE)
  cat("severity mean by group:\n")
  print(aggregate(severity ~ accession+treatment+timepoint, data=res, mean))
  cat("AUDPC mean by accession x treatment:\n")
  print(aggregate(AUDPC ~ accession+treatment, data=wide, function(x) round(mean(x,na.rm=TRUE),2)))
  cat("DONE severity\n")
}
