for /l %%i in (1,1,30) DO (for /l %%j in (1,1,30) DO (for %%k in (10)  DO CUDA_MBSO.exe -f input.txt -d 0 -func %%i -id %%k -ip 50 -r %%j -c 5))

for /l %%i in (1,1,30) DO (for /l %%j in (1,1,30) DO (for %%k in (10,30,50)  DO CUDA_MBSO.exe -f input.txt -d 0 -func %%i -id %%k -ip 50 -r %%j -c 5))

for /l %%i in (1,1,30) DO (for /l %%j in (1,1,30) DO (for %%k in (10,30,50)  DO CUDA_MBSO.exe -f input.txt -d 0 -func %%i -id %%k -ip 100 -r %%j -c 5))

