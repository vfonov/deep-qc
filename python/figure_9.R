suppressPackageStartupMessages(library(tidyverse))
suppressPackageStartupMessages(library(jsonlite))
suppressPackageStartupMessages(library(gridExtra))
# we only need resnet-18 with reference, best TNR results

cv_cls<-data.frame()
models=c('r18')
# all these are pretrained
for(lr_ in c('0.0001')) {
    pfx=paste0('cls/lr_',lr_,'_pre/')
    for(v in models) {
        for(r in c('_ref')) {
            for(f in seq(0,7)) {
                fn=paste0(pfx,'model_',v,r,'/log_',f,'_8.json')
                if(file.exists(fn)) {
                rrr=fromJSON(fn)

                fold<-as.data.frame(rrr$testing_best_tnr$detail) %>% mutate(fold=f)

                cv_cls<-bind_rows(cv_cls,fold)
                }
            }
        }
    }
}


cv_dist<-data.frame()
for(lr_ in c('0.0001')) {
    pfx=paste0('dist/lr_',lr_,'_pre/')
    for(v in models){
        for(r in c('_ref')) {
            for(f in seq(0,7)) {
                fn=paste0(pfx,'model_',v,r,'/log_',f,'_8.json')
                if(file.exists(fn)) {
                rrr=fromJSON(fn)

                #final<-as.data.frame(rrr$testing_final$details)
                best<-as.data.frame(rrr$testing_best_loss$details)
                # calculate AUC, TODO: calculate ACC?
                #r_final<-roc(final$labels, final$preds, stratified=F, auc=T)
                #r_best<-roc(best$labels, best$preds, stratified=F, auc=T)

                fold<-as.data.frame(rrr$testing_best_loss$details)%>% mutate(fold=f)
                cv_dist<-bind_rows(cv_dist,fold)
                }
            }
        }
    }
}

#cv<-cv%>%mutate(status=if_else(labels==0,'Fail','Pass'))

### analyze distance distributions
con<-DBI::dbConnect(RSQLite::SQLite(), "../data/qc_db.sqlite3")

# CREATE TABLE IF NOT EXISTS xfm_dist(variant,cohort,subject,visit,pass,lin,rx,ry,rz,tx,ty,tz,sx,sy,sz)
# right now drop distance augmented data
par<-DBI::dbReadTable(con,"xfm_dist") %>% filter(variant!='dist') %>%
  mutate(ids=paste(variant,cohort,subject,visit,sep=':'))

cv_cls_<-inner_join(cv_cls, par, by='ids') %>% 
 mutate(
    manual_qc=if_else(labels==0,'Fail','Pass'),
    auto_qc=case_when(
             (labels==1 & preds==1) ~ 'True positive',
             (labels==0 & preds==0) ~ 'True negative',
             (labels==0 & preds==1) ~ 'False positive',
             (labels==1 & preds==0) ~ 'False negative'
    )
 ) %>% mutate(auto_qc=as.factor(auto_qc),manual_qc=as.factor(manual_qc))


cv_dist_<-inner_join(cv_dist, par, by='ids') %>% 
 mutate(
    manual_qc=if_else(labels==0,'Fail','Pass')
 ) %>% mutate(manual_qc=as.factor(manual_qc))


dist_fit<-lm(lin~preds,data=cv_dist_)
dist_fit_r2=summary(dist_fit)$r.squared
dist_fit_r2_s=paste('r^2 : ',format(dist_fit_r2,digits=3))
dist_fit_slope=coef(dist_fit)[2]
dist_fit_intercept=coef(dist_fit)[1]


# print stats
print("Manual QC stats")
print(cv_cls_%>%group_by(manual_qc)%>%summarize(mean=mean(lin),sd=sd(lin)))

print("DARQ Resnet-18 QC stats")
print(cv_cls_%>%group_by(auto_qc)%>%summarize(mean=mean(lin),sd=sd(lin)))

#    theme(axis.text.x=element_text(angle=45,vjust=0.3))+

p1<-ggplot(cv_cls_,aes(y=lin, x=manual_qc))+
    theme_bw(base_size = 16)+
    theme(
      legend.position = c(.95, .50),
      legend.justification = c("right", "top"),
      legend.box.just = "right",
      legend.margin = margin(6, 6, 6, 6)
    )+
    geom_violin()+ylim(0,100)+
    xlab('')+ylab('Silver standard distance (mm)')+
    ggtitle("Manual QC")

p2<-ggplot(cv_cls_,aes(y=lin, x=auto_qc))+
    theme_bw(base_size = 16)+
    theme(
      legend.position = c(.95, .50),
      legend.justification = c("right", "top"),
      legend.box.just = "right",
      legend.margin = margin(6, 6, 6, 6)
    )+
    geom_violin()+ylim(0,100)+
    xlab('')+ylab('')+
    ggtitle("DARQ QCResNET-18")

p3<-ggplot(cv_dist_,aes(y=lin, x=preds))+
    theme_bw(base_size = 16)+
    theme(
      legend.position = c(.95, .50),
      legend.justification = c("right", "top"),
      legend.box.just = "right",
      legend.margin = margin(6, 6, 6, 6)
    )+
    geom_point(alpha=0.7)+
    coord_fixed(xlim=c(0,100),ylim=c(0,100))+
    geom_abline(slope=dist_fit_slope,intercept=dist_fit_intercept,lty=2,size=2,col='red')+
    geom_abline(slope=1,intercept=0,lty=2,size=1,col='blue',alpha=0.8)+
    annotate("label",x=25,y=50,label=dist_fit_r2_s, col='red', size=10, parse=T)+
    xlab('Prediction (mm)')+ylab('')+
    ggtitle("DARQ DistResNET-18")


png("Figure_9_distance_r18.png", 
    width=30, height=10, res=200, units = "in", 
    pointsize = 12, type='cairo', antialias = "default")

grid.arrange(p1,p2,p3,nrow=1)

q()
