dl()
{  
	  for (( num=$1; num<=$2; num++))
		    do
			      dmux job $num get-results
			        done
			}

dl 124865 124892 
