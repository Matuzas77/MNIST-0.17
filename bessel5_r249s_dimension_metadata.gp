\\ BESSEL5 R249S exact characteristic-zero dimension metadata.
\\ PARI mfinit space flags: 0=new, 1=cuspidal, 2=old, 3=Eisenstein, 4=full.

default(parisizemax, 4000000000);
default(parisize, 536870912);

check_ok(c,msg)=if(c==0,error(msg));

one_level(N)=
{
  my(k=3,chi=-3,dnew,dcusp,dold,deis,dfull);
  print("LEVEL_BEGIN N=",N);
  dnew=mfdim([N,k,chi],0);
  dcusp=mfdim([N,k,chi],1);
  dold=mfdim([N,k,chi],2);
  deis=mfdim([N,k,chi],3);
  dfull=mfdim([N,k,chi],4);
  print("DIMENSIONS N=",N," new=",dnew," cusp=",dcusp," old=",dold," eisenstein=",deis," full=",dfull);
  check_ok(dcusp==dnew+dold,Str("cusp decomposition failed at N=",N));
  check_ok(dfull==dcusp+deis,Str("full decomposition failed at N=",N));
  write("r249_dimensions/metadata.txt",Str("N=",N," new=",dnew," cusp=",dcusp," old=",dold," eisenstein=",deis," full=",dfull));
  print("LEVEL_END N=",N);
  return([dnew,dcusp,dold,deis,dfull])
};

main()=
{
  my(d38400,d19200,d7680);
  print("BESSEL5_R249S_DIMENSION_METADATA_START");
  d38400=one_level(38400);
  d19200=one_level(19200);
  d7680=one_level(7680);
  check_ok(d38400[1]==2432,Str("unexpected level-38400 new dimension ",d38400[1]));
  write("r249_dimensions/result.gp",[d38400,d19200,d7680]);
  write("r249_dimensions/SUCCESS","PASS");
  print("SELF_CHECK=PASS");
  print("BESSEL5_R249S_DIMENSION_METADATA_END");
  return(d38400)
};

main();
quit;
