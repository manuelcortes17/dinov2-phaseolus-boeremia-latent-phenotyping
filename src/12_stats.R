# 12_stats.R — Multivariate + hierarchical statistics (leakage-safe). No sink().
suppressWarnings(suppressMessages({library(vegan); library(lme4); library(lmerTest); library(emmeans)}))
for(loc in c("C.UTF-8","en_US.UTF-8")) if(!inherits(try(Sys.setlocale("LC_ALL",loc),silent=TRUE),"try-error")) break
set.seed(42)
OUT<-"/home/claude/proj/outputs/stats"; dir.create(OUT,showWarnings=FALSE,recursive=TRUE)
say<-function(...) { cat(...,"\n"); flush.console() }

U <- read.csv("/home/claude/proj/outputs/trajectories/unit_level.csv", stringsAsFactors=FALSE)
UV <- as.matrix(read.csv("/home/claude/proj/outputs/trajectories/unit_vectors_csv.csv", header=FALSE))
U$accession<-factor(U$accession); U$treatment<-factor(U$treatment); U$timepoint<-factor(U$timepoint)
D <- dist(UV)
say("unit-level rows:", nrow(U), "dims:", ncol(UV))

say("\n### PERMANOVA (adonis2, 999 perm, by=terms) ###")
set.seed(1); t<-system.time(pm <- adonis2(D ~ accession*treatment*timepoint, data=U, permutations=999, by="terms"))
print(pm); say("elapsed",t[3],"s")
write.csv(as.data.frame(pm), file.path(OUT,"permanova.csv"))

say("\n### PERMDISP betadisper by timepoint (inoculated) ###")
inoc <- U$treatment=="inoculated"; Di <- dist(UV[inoc,]); tpi <- droplevels(U$timepoint[inoc])
bd <- betadisper(Di, tpi); print(anova(bd))
md<-tapply(bd$distances, tpi, mean); print(md)
pd<-permutest(bd, permutations=999); print(pd$tab)
write.csv(data.frame(timepoint=names(md), mean_dist=as.numeric(md)), file.path(OUT,"permdisp.csv"), row.names=FALSE)

say("\n### LMM log1p(severity) ~ accession*timepoint + (1|unit) [inoculated] ###")
sev <- read.csv("/home/claude/proj/outputs/r4pde/severity_per_image.csv", stringsAsFactors=FALSE)
sev <- sev[sev$treatment=="inoculated" & is.finite(sev$severity),]
sev$logsev<-log1p(sev$severity); sev$accession<-factor(sev$accession); sev$timepoint<-factor(sev$timepoint)
m1 <- lmer(logsev ~ accession*timepoint + (1|biological_unit_id), data=sev)
print(anova(m1))
em <- emmeans(m1, ~ accession | timepoint); print(em); print(pairs(em))
write.csv(as.data.frame(summary(m1)$coefficients), file.path(OUT,"lmm_severity_coef.csv"))
write.csv(as.data.frame(em), file.path(OUT,"lmm_severity_emmeans.csv"))
write.csv(as.data.frame(pairs(em)), file.path(OUT,"lmm_severity_contrasts.csv"))

say("\n### LMM spatial metrics ~ accession*timepoint + (1|unit) [inoculated] ###")
sp <- read.csv("/home/claude/proj/outputs/spatial/spatial_per_image.csv", stringsAsFactors=FALSE)
sp <- sp[sp$treatment=="inoculated" & is.finite(sp$morans_I),]
sp$accession<-factor(sp$accession); sp$timepoint<-factor(sp$timepoint)
spatres<-list()
for(v in c("morans_I","lesion_fraction","heterogeneity","mean_lesion_score")){
  f<-as.formula(paste0(v," ~ accession*timepoint + (1|biological_unit_id)"))
  mm<-lmer(f, data=sp); a<-anova(mm); say("---",v,"---"); print(a)
  spatres[[v]]<-a
}
saveRDS(spatres, file.path(OUT,"spatial_lmm.rds"))
say("\nSTATS DONE")
