dl()
{  
  for (( num=$1; num<=$2; num++))
  do
  dmux job $num get-results
  done
}

dkill()
{
  for (( num=$1; num<=$2; num++))
  do
  dmux job $num kill
  done
}

#dl 52188 52192
#dl 52874 52887
#dl 52891 52926
#dl 52928 52976
#dl 53112 53143 
#dl 55951 56419
#dl 56150 56419
#dl 56100 56050
#dl 55994 56019
#dl 56679 56780
#dl 56779 57204
#dl 57204 57216
#dl 57418 57490
#dl 57491 57562
#dl 57563 57635
#dl 57636 57742
#dl 58005 58223
#dl 58223 58452
#dl 58696 58725
#dl 58726 58946
#dl 58946 59488
#dl 59725 59778
#dl 59779 59929
#dl 59930 60170
#dl 60999 61012
#dl 61210 61900
#dl 66267 68082
#dl 66636 67079 &
#dl 67079 68000 &
#dl 68000 68082 &
#dl 67800 68000 &
#dl 67600 67800 &
#dl 67400 67600 &
#dl 67200 67400 &
#dl 67000 67200 &
#sleep 10800;
#dl 76070 78251;
#dl 78251 79000;
#kill 76070 83889
#dl 85623 85944
#dl 85945 86040
#dl 86041 86080
#dl 86081 86161
#dl 86161 86265
#dl 86266 86374
#dl 86375 86576
#dl 86577 86631
#dl 86632 86831
#dl 86900 87027
#dl 87028 87198
#dl 87199 87308
#dl 88773 88852
#dl 89003 89022
#dl 89022 89476
#kill 89600 89693
#dl 91519 91524
#dl 91855 91860
#dl 99511 99556
#dl 107736 107744
#dl 108788 109203
#dl 115580 115788
#dl 122200 122422

#dl 133126 133155
#dl 133595 133604
#dl 133640 133659
#dl 134185 134204
#dl 134328 134337
# mtn car
#dl 134963 134976

#dl 136137 136156
#dl 136867 136876
#dl 139134 139143

#dkill 139207 139360
#dkill 141093 141107
#dkill 141123 141127
#dkill 141188 141202
#dkill 141712 141717
#dkill 141316 141319
#dkill 141108 141132

#dkill 144452 144491
#dl 145060 145077
#dkill 145167 145184
#dkill 144452 144491
#dkill 145213 145265
#dkill 145268 145313
#dkill 198939 198956:wq
dkill 199195 199203

