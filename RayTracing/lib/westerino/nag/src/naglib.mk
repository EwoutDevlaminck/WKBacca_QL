# firts remove the library when it exists!
rm -f libnag.a

# first compile all .f source files
ifort *.f -c  -O3 -r8 -align dcommons -save

#Now create the library using the 'ar' command
echo Creating the library libnag.a
ar vq libnag.a *.o
