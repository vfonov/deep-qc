##########
#
# Show statistics on one CV fold
#
#
#
##########
suppressPackageStartupMessages(library(tidyverse))
suppressPackageStartupMessages(library(jsonlite))
suppressPackageStartupMessages(library(gridExtra))
library(pROC)


online<-data.frame()
final_summary<-data.frame()
final_detail<-data.frame()

# best_summary<-data.frame()
# best_detail<-data.frame()


models=c('r18') # 'r152','r101','r50','r34',

for(m in models)
{
    # validation stats
    r<-fromJSON(paste0('model_dist_',m,'_ref_long/log_0_8.json')) # _unlimited
    val<-r$validation
    # convert minibatches to fractional epoch
    sc<-val%>%summarize(epoch=max(epoch),minibatch=max(ctr))%>%transmute(sc=epoch/minibatch)

    val<-val%>%mutate(f_epoch=ctr*sc$sc)

    val$model=m
    online<-bind_rows(online,val)

    final_<-as.data.frame(r$testing_final$summary) %>% mutate(model=m,f_epoch=max(val$f_epoch))
    final_summary<-bind_rows(final_summary,final_)

    final_detail<-bind_rows(final_detail,as.data.frame(r$testing_final$detail) %>% mutate(model=m))

    # best_<-as.data.frame(r$testing_best_loss$summary) %>% mutate(model=m, f_epoch=max(val$f_epoch))
    # best_summary<-bind_rows(best_summary,best_)

    # best_detail<-bind_rows(best_detail,as.data.frame(r$testing_best_loss$detail) %>% mutate(model=m))
}

online<-online %>%  mutate(model=factor(model,levels=models))
# final
final_summary<-final_summary %>%  mutate(model=factor(model,levels=models))
final_detail<-final_detail %>%  mutate(model=factor(model,levels=models))
# best
# best_summary<-best_summary %>%  mutate(model=factor(model,levels=models))
# best_detail<-best_detail %>%  mutate(model=factor(model,levels=models))
###

### analyze distributions
con<-DBI::dbConnect(RSQLite::SQLite(), "../data/qc_db.sqlite3")

# CREATE TABLE IF NOT EXISTS xfm_dist(variant,cohort,subject,visit,pass,lin,rx,ry,rz,tx,ty,tz,sx,sy,sz)
par<-DBI::dbReadTable(con,"xfm_dist") %>% 
  mutate(variant=as.factor(variant),cohort=as.factor(cohort)) %>%
  mutate(training=as.factor(if_else(variant=='dist','Training','Testing')))

p1<-ggplot(par,aes(y=lin, x=training))+
    theme_bw(base_size = 28)+
    theme(
      legend.position = c(.95, .50),
      legend.justification = c("right", "top"),
      legend.box.just = "right",
      legend.margin = margin(6, 6, 6, 6)
    )+
    geom_violin()+
    stat_summary(
        fun.data = "median_hilow", 
        geom = "pointrange", color = "black"
        )+
    ylim(0,100)+
    xlab('')+ylab('Distance (mm)')+
    ggtitle("Distance distributions")


online_m <- online 

p2<-ggplot(online_m, aes(y=loss, x=f_epoch))+
    theme_bw(base_size = 28)+
    theme(
      legend.position = c(.95, .50),
      legend.justification = c("right", "top"),
      legend.box.just = "right",
      legend.margin = margin(6, 6, 6, 6)
    )+
    geom_line()+
    xlab('Epoch')+ylab('Mean Square Error')+
    guides(color=guide_legend(title=""))+
    ggtitle("Online validation")


fit<-data.frame()
fit2<-data.frame()

models_<-c()
models2_<-c()

for(s in models) {
  ss<-final_detail %>% filter(model==s)
  m<-lm(dist~preds,data=ss)
  r2=summary(m)$r.squared
  r2_s=paste(s,'r2:',format(r2,digits=3))

  fit<-bind_rows(fit,data.frame(
      slope=coef(m)[2],intercept=coef(m)[1],
      r2=r2,
      model_=r2_s,
      model=s))
  models_<-c(models_,r2_s)

  # # best
  # ss<-best_detail %>% filter(model==s)
  # m<-lm(dist~preds,data=ss)
  # r2=summary(m)$r.squared
  # r2_s=paste(s,'r2:',format(r2,digits=3))

  # fit2<-bind_rows(fit2,data.frame(
  #     slope=coef(m)[2],intercept=coef(m)[1],
  #     r2=r2,
  #     model_=r2_s,
  #     model=s))
  # models2_<-c(models2_,r2_s)
}

final_detail<-final_detail%>%mutate(model_=factor(model,levels=models,labels=models_))
# best_detail<-best_detail%>%mutate(model_=factor(model,levels=models,labels=models2_))

p3<-ggplot(data=final_detail,aes(y=dist,x=preds, color=model_))+
  theme_bw(base_size = 28)+
  theme(
    legend.position = c(.95, .50),
    legend.justification = c("right", "top"),
    legend.box.just = "right",
    legend.margin = margin(6, 6, 6, 6)
    )+
  geom_point() + 
  geom_abline(data=fit, aes(slope=slope,intercept=intercept,color=model_))+
  coord_fixed(xlim=c(0,100),ylim=c(0,100))+
  geom_abline(slope=1.0,intercept=0.0, lty=2, col='black')+
  guides(color=guide_legend(title=""))+
  xlab("Prediction (mm)")+ylab("Silver standard (mm)")+
  ggtitle("Distance prediction, final model")

# p3<-ggplot(data=best_detail,aes(y=dist,x=preds, color=model_))+
#   theme_bw(base_size = 14)+
#   theme(
#     legend.position = c(.95, .50),
#     legend.justification = c("right", "top"),
#     legend.box.just = "right",
#     legend.margin = margin(6, 6, 6, 6)
#     )+
#   geom_point() + 
#   geom_abline(data=fit2, aes(slope=slope,intercept=intercept,color=model_))+
#   coord_fixed(xlim=c(0,100),ylim=c(0,100))+
#   geom_abline(slope=1.0,intercept=0.0, lty=2, col='black')+
#   guides(color=guide_legend(title=""))+
#   ggtitle("Prediction, best MSE error ")

auc<-data.frame()
auc_s<-data.frame()
models_<-c()


mylogit.inv<-function(x) 1/(1+exp(-x)) 

for(s in models) {
    ss<-final_detail%>%filter(model==s)
    r<-roc(ss$labels, mylogit.inv(ss$preds-10), stratified=T,auc=T) # 
    a<-paste(s,' estimate AUC:',format(r$auc,digits=3))
    models_<-c(models_,a)

    auc<-bind_rows(auc,data.frame(
                         TPR=r$sensitivities,
                         FPR=1.0-r$specificities
                        ) %>% mutate(model=a)
                   )

    auc_s<-bind_rows(auc_s, data.frame(auc=as.numeric(r$auc), model=s))

    ## 
    # ss<-best_detail%>%filter(model==s)
    # r<-roc(ss$labels, mylogit.inv(ss$preds-10), stratified=T,auc=T) # 
    # a<-paste(s,'best auc:',format(r$auc,digits=3))
    # models_<-c(models_,a)

    # auc<-bind_rows(auc,data.frame(
    #                      tpr=r$sensitivities,
    #                      fpr=1.0-r$specificities
    #                     ) %>% mutate(model=a)
    #                )

    # auc_s<-bind_rows(auc_s, data.frame(auc=r$auc, model=s))
}
# reference

ss<-final_detail%>%filter(model=='r18')
rref<-roc(ss$labels, mylogit.inv(ss$dist-10), stratified=T,auc=T) # 
a<-paste('Silver standard','AUC:',format(rref$auc,digits=3))
models_<-c(models_,a)

auc<-bind_rows(auc,data.frame(
                      TPR=rref$sensitivities,
                      FPR=1.0-rref$specificities
                    ) %>% mutate(model=a)
                )

auc_s<-bind_rows(auc_s, data.frame(auc=as.numeric(rref$auc), model='Silver standard'))

auc<-auc %>%  mutate(model=factor(model,levels=models_))
auc_s<-auc_s %>%  mutate(model=factor(model,levels=c(models,'Silver standard')))

print(auc_s)

p4<-ggplot(data=auc,aes(y=TPR,x=FPR, color=model))+
  theme_bw(base_size = 28)+
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
  ggtitle("ROC on distance for predicting Pass/Fail")


png("Figure_6_distance_training.png", width=20, height=20, res=200, units = "in", pointsize = 12, type='cairo', antialias = "default")

grid.arrange(p1,p2,p3,p4)
