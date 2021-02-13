#include<stdio.h>
#include<stdlib.h>

void loc(float *x,int nl);
void read(FILE *fp,char name[],float x[],int nl);


void main()
{
	int il,nl=2700*25+1;

	float *x,*z,*rho,*vp,*vs;

	FILE *fp;

	x=(float*)calloc(nl,sizeof(float));
	z=(float*)calloc(nl,sizeof(float));
	rho=(float*)calloc(nl,sizeof(float));
	vp=(float*)calloc(nl,sizeof(float));
	vs=(float*)calloc(nl,sizeof(float));

	//loc(x,nl);
	//loc(z,nl);
	//loc(rho,nl);
	//loc(vp,nl);
	//loc(vs,nl);
	read(fp,"proc000000_x.bin",x,nl);
	read(fp,"proc000000_z.bin",z,nl);	
	read(fp,"proc000000_vp.bin",rho,nl-1);
	read(fp,"proc000000_vp.bin",vp,nl-1);
	read(fp,"proc000000_vs.bin",vs,nl-1);

	//fp=fopen("proc000000_x.bin","rb");
	//fread(x,sizeof(float),nl,fp);
	//fclose(fp);

	//fp=fopen("newx.bin","wb");
	//fwrite(x,sizeof(float),nl,fp);
	//fclose(fp);

	fp=fopen("proc000000_rho_kappa_mu_kernel.dat","w");
	for(il=1;il<nl;il++){
		fprintf(fp,"%f\t%f\t%f\t%f\t%f\n",x[il],z[il],vp[il-1],vp[il-1],vs[il-1]);
	}
	fclose(fp);


}

void loc(float *x,int nl){
	x=(float*)calloc(nl,sizeof(float));
}

void read(FILE *fp,char name[],float x[],int nl){
	fp=fopen(name,"rb");
	fread(x,sizeof(float),nl,fp);
	fclose(fp);
}
