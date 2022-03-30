##########
#
# Show statistics on one CV fold
#
#
#
##########
library(MASS)
suppressPackageStartupMessages(library(tidyverse))
suppressPackageStartupMessages(library(jsonlite))
suppressPackageStartupMessages(library(gridExtra))
library(pROC)


online<-data.frame()
final_summary<-data.frame()
final_detail<-data.frame()

# best_summary<-data.frame()
# best_detail<-data.frame()

### analyze distributions
con<-DBI::dbConnect(RSQLite::SQLite(), "../data/qc_db.sqlite3")

# CREATE TABLE IF NOT EXISTS xfm_dist(variant,cohort,subject,visit,pass,lin,rx,ry,rz,tx,ty,tz,sx,sy,sz)
par<-DBI::dbReadTable(con,"xfm_dist") %>%
  filter(variant!='dist') %>% 
  mutate(variant=as.factor(variant),
         cohort=as.factor(cohort),
         pass=pass=="TRUE",
         pass_t=factor(pass,levels=c(F,T),labels=c("Fail","Pass")))

my.glm<-glm(pass~lin, data = par, family = binomial)

# optimal threshold
threshold=dose.p(my.glm, p = 0.5)[[1]]
print("Threshold")
print(threshold)



p1<-ggplot(par,aes(y=lin, x=pass_t))+
    theme_bw(base_size = 28)+
    theme(
      legend.position = c(.95, .50),
      legend.justification = c("right", "top"),
      legend.box.just = "right",
      legend.margin = margin(6, 6, 6, 6)
    )+
    geom_hline(yintercept=threshold,color="red",
               lty=2, size=1.5, alpha=0.9)+
    geom_text(y=threshold+1, x=0.5, color="red", 
             label=round(threshold,2),
             vjust="bottom", size=4)+
    geom_violin(alpha=0.7)+
    stat_summary(
        fun.data = "median_hilow", 
        geom = "pointrange", color = "black"
        )+
    ylim(0,100)+
    xlab('Manual QC')+ylab('Distance (mm)')+
    ggtitle("Distance distributions")


par$p=predict(my.glm,type = "response")
r<-roc(par$pass, par$p, stratified=T,auc=T) # 


auc<-data.frame(      TPR=r$sensitivities,
                      FPR=1.0-r$specificities
                    )

p2<-ggplot(data=auc,aes(y=TPR,x=FPR))+
  theme_bw(base_size = 22)+
  theme(
    legend.position = c(.95, .50),
    legend.justification = c("right", "top"),
    legend.box.just = "right",
    legend.margin = margin(6, 6, 6, 6)
    )+
  guides(color=guide_legend(title=""))+
  geom_line()+
  coord_fixed()+
  geom_abline(slope=1.0,intercept=0.0, lty=2, col='black')+
  geom_text(y=0.75, x=0.25, color="red", size=4,
             label=sprintf("AUC=%0.2f",as.numeric(r$auc)))+
  ggtitle("ROC on distance for predicting Pass/Fail")


png("Figure_XX_distance_training.png", width=20, height=10, 
  res=200, units = "in", pointsize = 12, type='cairo', antialias = "default")

grid.arrange(p1,p2,ncol=2)
#print(p1)