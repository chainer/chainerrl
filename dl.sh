dl()
{  
	  for (( num=$1; num<=$2; num++))
		    do
			      dmux job $num get-results
			        done
			}

dl 125407 125469
