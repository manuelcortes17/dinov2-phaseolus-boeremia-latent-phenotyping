# 10b_sevpanel.R — R4PDE-style severity panel: several leaves, original vs segmented.
suppressWarnings(suppressMessages({library(jpeg)}))
for(loc in c("C.UTF-8","en_US.UTF-8")) if(!inherits(try(Sys.setlocale("LC_ALL",loc),silent=TRUE),"try-error")) break
PAL <- list(
  healthy = rbind(c(.25,.45,.15),c(.35,.55,.20),c(.20,.40,.12),c(.45,.60,.25),c(.30,.50,.18),c(.15,.35,.10),c(.40,.58,.22)),
  lesion  = rbind(c(.78,.78,.22),c(.68,.62,.16),c(.82,.80,.38),c(.72,.68,.20),c(.50,.40,.20),c(.40,.30,.15),c(.60,.48,.28),
                  c(.34,.25,.13),c(.55,.45,.18),c(.66,.55,.30),c(.28,.20,.12),c(.22,.16,.10),c(.32,.24,.14),c(.24,.18,.11)),
  backg   = rbind(c(.80,.82,.82),c(.70,.72,.72),c(.88,.90,.90),c(.60,.62,.63),c(.75,.78,.79),c(.66,.69,.70)),
  dark    = rbind(c(.05,.05,.05),c(.14,.14,.14),c(.02,.02,.02),c(.20,.20,.21)),
  light   = rbind(c(.96,.96,.96),c(1,1,1),c(.90,.90,.92),c(.98,.97,.95)))
CLASSES<-names(PAL); PL<-lapply(PAL,function(x) convertColor(x,"sRGB","Lab"))
classify<-function(img){h<-dim(img)[1];w<-dim(img)[2];px<-matrix(img,ncol=3);lab<-convertColor(px,"sRGB","Lab")
  best<-rep(1L,nrow(lab));bd<-rep(Inf,nrow(lab))
  for(ci in seq_along(CLASSES)){P<-PL[[ci]];dm<-rep(Inf,nrow(lab))
    for(j in seq_len(nrow(P))) dm<-pmin(dm,(lab[,1]-P[j,1])^2+(lab[,2]-P[j,2])^2+(lab[,3]-P[j,3])^2)
    u<-dm<bd;best[u]<-ci;bd[u]<-dm[u]};matrix(best,nrow=h)}
idx<-read.csv("/home/claude/proj/outputs/index/master_index.csv",stringsAsFactors=FALSE)
sev<-read.csv("/home/claude/proj/outputs/r4pde/severity_per_image.csv",stringsAsFactors=FALSE)
cand<-merge(idx[idx$is_augmented=="False",],sev[,c("image_id","severity")],by="image_id")
cand<-cand[cand$accession=="ASC 160"&cand$treatment=="inoculated"&!is.na(cand$severity),]
pick<-function(tg){ cand$img_path[order(abs(cand$severity-tg))][1] }
picks<-c(pick(9),pick(22),pick(42)); blue<-c(0.16,0.45,0.72)
segment_one<-function(p){img<-readJPEG(p);cls<-classify(img);h<-dim(img)[1];w<-dim(img)[2]
  lm<-cls %in% c(which(CLASSES=="healthy"),which(CLASSES=="lesion"))
  oimg<-img;for(k in 1:3){ch<-oimg[,,k];ch[!lm]<-blue[k];oimg[,,k]<-ch}
  hh<-cls==which(CLASSES=="healthy");ll<-cls==which(CLASSES=="lesion")
  oR<-matrix(blue[1],h,w);oG<-matrix(blue[2],h,w);oB<-matrix(blue[3],h,w)
  oR[hh]<-.18;oG[hh]<-.80;oB[hh]<-.20;oR[ll]<-0;oG[ll]<-0;oB[ll]<-0
  seg<-array(0,c(h,w,3));seg[,,1]<-oR;seg[,,2]<-oG;seg[,,3]<-oB
  sv<-100*sum(ll)/sum(hh|ll);list(o=oimg,s=seg,sv=sv)}
R<-lapply(picks,segment_one)
png("/home/claude/proj/outputs/supp_figures/sev_calc_panel.png",width=2700,height=900,res=165)
par(mfrow=c(1,6),mar=c(0.3,0.3,3.6,0.3),bg="white")
for(r in R){
  plot(as.raster(r$o)); title("original",cex.main=2.2,font.main=2,line=1.6)
  mtext(sprintf("severity = %.0f%%",r$sv),side=3,line=0.2,cex=1.35,col="#B2182B",font=2)
  plot(as.raster(r$s)); title("segmented",cex.main=2.2,font.main=2,line=1.6)
  mtext("healthy / lesion",side=3,line=0.2,cex=1.2,col="#444444")
}
dev.off(); cat("severities:",sapply(R,function(x)round(x$sv,0)),"\n")
