library(tidyverse)
library(DBI)
# install.packages("RSQLite")

con <- dbConnect(RSQLite::SQLite(), "data/qc_db.sqlite3")

qc<-dbReadTable(con, "qc_all")
qc<-qc %>% mutate(pass=(pass=="TRUE"))

qc_sum<-summarise(group_by(qc,variant,cohort), passed=sum(pass)*100/n(),failed=sum(!pass)*100/n(),total=n())
write.csv( t(qc_sum),"results/qc_sum.csv",quote=F)


print(summarise(qc,passed=sum(pass)*100/n(),failed=sum(!pass)*100/n(),total=n()))
