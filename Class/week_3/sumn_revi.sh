#!/bin/bash
echo "------SUM OF N NUMBERS AND FACTORIAL IN SHELL SCRIPT-----"
echo -n "Enter nth number's value : "
read digit
t=1
total=0
fact=1
while test $t -le $digit
do
	# for sum
	total=`expr $total + $t`
	# for factorial
	fact=$((fact*t))
	
	t=`expr $t + 1`	
done
echo "sum of $digit: $total "

echo "factorial of $digit: $fact "


