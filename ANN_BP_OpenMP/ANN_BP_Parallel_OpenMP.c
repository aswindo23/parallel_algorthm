#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <pthread.h>
#include <omp.h>





/// -+          Header              +-
///------------------------------------------------------

    double input[1501][10];           /// Var Input dataset
    double net_h[1501][5];           /// Var Network of Hidden Layer     net_h[hidden layer N ][ input N ]
    double output_h[1501][5];        /// Var Output of Hidden Layer      output_h[hidden layer N ][ output N (net_h) ]
    double net_o[1501][5];           /// Var Network of Output Layer     net_h[Output layer N ][ output hidden layer(input) N ]
    double output_o[1501][5];        /// Var Network of Output from Hiden layer Output output_o[Output layer N ][ net_o N ]
    double error[1501][10];           /// Var Error   output layer        Error[Error output][itereasi (form output layer)]
    double Total_error[1501];       /// Var Total Error                 Total_error[Hidden Layer]
    double TranningData[1501][31];         /// Var Compute Trainning Dataset
    double weight[1501][12];          /// Var Weight
    double weightTebakan[20];
    double deltha_o[1501][2];
    double devarative_o[1501][2];
    double deltha_h[1501][2];
    double devarative_h[1501][2];


int n,N,nN;


/// BackPropagation Network
void getDataTrainning_fromFile(char* file);
double createNetwork(double w1, double w2, double w3, double i1,double i2, double b);
double output(double net);
double Error(double target, double out);
void Forward_Pass();
void Backward_Pass();
double dho_output(double out);
double compute_deltha(double target, double out);
double devarative_function_o(double deltha_i, double out );
double compute_deltha_h(double delthao1,double deltha_o2, double w1, double w2, double out);
double devarative_function_h(double deltha_h,double input);
void update_weigth(double alpha);
double update_weigth_function(double W,double learningrate, double devarative_f );
double sumError(double target, double out, double out2);
double mean_se();

double createNetwork(double w1, double w2, double w3, double i1,double i2, double b){
    return ((w1*i1)+(w2*i2)+(w3*b));
}

double output(double net){
    return (1/(1+exp(-net)));
}

double Error(double target, double out){
    return (0.5*pow((target-out),2));
}


double compute_deltha(double target, double out){
    return (-(target-out)*dho_output(out));
}

double devarative_function_o(double deltha_i, double out ){
    return -deltha_i*out;
}

double dho_output(double out){
    double o ;
    o = (double) (1-out);
    return (out*o);
}

double compute_deltha_h(double deltha_o1,double deltha_o2, double w1, double w2, double out){
    return (((deltha_o1*w1)+(deltha_o2*w2))*dho_output(out));
}

double devarative_function_h(double deltha_h,double input){
    return -deltha_h*input;
}


double update_weigth_function(double W,double learningrate, double devarative_f ){
    return (W-(learningrate*devarative_f));

}
double sumError(double target, double out, double out2){
    return  (0.5*(pow((target-out),2)) + 0.5*(pow((target-out2),2))) - (atan(target-out) + atan(target-out2 ) );
}

double mean_se(){
    int i,j;
    double sum =0;
        #pragma omp parallel num_threads(8)
        {
                #pragma omp for schedule(dynamic,1)
                for(i=1; i<=nN; i++){
                    sum += sumError(input[i][4],output_o[i][1],output_o[i][2]);
                }
        }


    return sum/(2*nN);
}

void Forward_Pass(){  /// Trainning with Back Propagetion

    int i,j,e;
        printf("----------------------------------------------------------------------------------------------------------------------------------------------------------------\n");
        printf(" %-11s | %-12s | %-12s | %-12s | %-12s | %-12s | %-12s | %-12s | %-12s | %-12s | %-12s |","Net h1","Out h1","Net h2","Out h2","Net o1","Out o1","Net o2","Out o2","Eo1","Eo2","Etotal"); printf("\n");
        printf("----------------------------------------------------------------------------------------------------------------------------------------------------------------\n");

    double start = omp_get_wtime();

    #pragma omp parallel num_threads(8)
    {
    #pragma omp for schedule(dynamic,1)
    for(i=1; i<=nN; i++){
        for(j=1; j<=11; j++){

            switch(j){
                case 1  :  /// Generate Network from input to hidden layer 1
                        #pragma omp critical
                        TranningData[i][j] = createNetwork(weightTebakan[1],weightTebakan[2],weightTebakan[3],input[i][2],input[i][3],1);
                        net_h[i][1] = TranningData[i][j];  /// [i][1] => Network hidden layer 1
                    break;
                case 2  : /// Generate Output from input to hidden layer
                        #pragma omp critical
                        TranningData[i][j] = output(net_h[i][1]);
                        output_h[i][1] = TranningData[i][j];
                    break;
                case 3 : /// Generate Network from input to hidden layer 2
                        #pragma omp critical
                        TranningData[i][j] = createNetwork(weightTebakan[4],weightTebakan[5],weightTebakan[6],input[i][2],input[i][3],1);
                        net_h[i][2] = TranningData[i][j];  /// [i][1] => Network hidden layer 1
                    break;
                case 4 : /// Generate Output from input to hidden layer 2
                        #pragma omp critical
                        TranningData[i][j] = output(net_h[i][2]);
                        output_h[i][2] = TranningData[i][j];
                    break;
                case 5 : /// Generete Network from hidden layer to output layer
                        #pragma omp critical
                        TranningData[i][j] = createNetwork(weightTebakan[7],weightTebakan[8],weightTebakan[9],output_h[i][1],output_h[i][2],1);
                        net_o[i][1] =  TranningData[i][j];
                    break;
                case 6 : /// Generete Output from hidden layer to output layer
                        #pragma omp critical
                        TranningData[i][j] = output(net_o[i][1]);
                        output_o[i][1] =  TranningData[i][j];
                    break;
                case 7 : /// Generete Network from hidden layer to output layer
                        #pragma omp critical
                        TranningData[i][j] = createNetwork(weightTebakan[10],weightTebakan[11],weightTebakan[12],output_h[i][1],output_h[i][2],1);
                        net_o[i][2] =  TranningData[i][j];
                    break;
                case 8 : /// Generete Output from hidden layer to output layer
                        #pragma omp critical
                        TranningData[i][j] = output(net_o[i][2]);
                        output_o[i][2] =  TranningData[i][j];
                    break;
                case 9  : /// Calculate Error E01 from output H->O
                        #pragma omp critical
                        TranningData[i][j] = Error(input[i][4],output_o[i][1]);
                        error[i][1] =  TranningData[i][j];
                    break;
                case 10  : /// Calculate Error E01 from output H->O
                        #pragma omp critical
                        TranningData[i][j] = Error(input[i][4],output_o[i][2]);
                        error[i][2] =  TranningData[i][j];
                    break;
                case 11  : /// Total Error Eo1 Eo2 from output H->O
                        #pragma omp critical
                        TranningData[i][j] = error[i][1] + error[i][2];
                        Total_error[i] =  TranningData[i][j];
                    break;
            }
        }

        printf("Data ke-[%i] | %5.10lf | %5.10lf | %5.10lf | %5.10lf | %5.10lf | %5.10lf | %5.10lf | %5.10lf | %5.10lf | %5.10lf | %5.10lf | \n",i,TranningData[i][1],TranningData[i][2],TranningData[i][3],TranningData[i][4],TranningData[i][5],TranningData[i][6],TranningData[i][7],TranningData[i][8],TranningData[i][9],TranningData[i][10],TranningData[i][11]);
    }
    }


    double stop = omp_get_wtime() - start;
    printf("----------------------------------\n");
    printf("Waktu Eksekusi File    : %lf \n",stop);
    printf("----------------------------------\n");
    //getchar();
}

void Backward_Pass(){
    int i,j;

    printf("----------------------------------------------------------------------------------------------------------------------------------------------------------------\n");
    printf(" %-11s | %-12s | %-12s | %-12s | %-12s | %-12s | %-12s | %-12s | %-12s | %-12s | %-12s |","Deltha O1","Dev O1","Deltha O2","Dev O2","Delta H1","Dev H1","Delta H2","Dev H2","Eo1","Eo2","Etotal"); printf("\n");
    printf("----------------------------------------------------------------------------------------------------------------------------------------------------------------\n");

    double start2 = omp_get_wtime();
    #pragma omp parallel num_threads(8)
    {
    #pragma omp for schedule(dynamic,1)
    for(i=1; i<=nN; i++){
        for(j=12; j<=19; j++){
            switch(j){
                case 12 : /// Delta Output Layer 1

                        TranningData[i][j] = compute_deltha(input[i][4],output_o[i][1]);
                        deltha_o[i][1] =  TranningData[i][j];
                    break;
                case 13 : /// Compute Devarative function sigmoit from output layer 1
                        #pragma omp critical
                        TranningData[i][j] = devarative_function_o(deltha_o[i][1],output_o[i][1]);
                        devarative_o[i][1] = TranningData[i][j];
                    break;
                case 14 : /// Delta Output Layer 2
                        TranningData[i][j] = compute_deltha(input[i][4],output_o[i][2]);
                        deltha_o[i][2] =  TranningData[i][j];
                    break;
                case 15 : /// Compute Devarative function sigmoit from output layer 1
                        #pragma omp critical
                        TranningData[i][j] = devarative_function_o(deltha_o[i][2],output_o[i][2]);
                        devarative_o[i][2] = TranningData[i][j];
                    break;
                case 16 : /// Delta Hidden Layer 1
                        #pragma omp critical
                        TranningData[i][j] = compute_deltha_h(deltha_o[i][1],deltha_o[i][2],weightTebakan[7],weightTebakan[8],output_h[i][1] );
                        deltha_h[i][1] = TranningData[i][j];
                    break;
                case 17 : /// Compute Devarative function sigmoit from hidden layer 1
                        #pragma omp critical
                        TranningData[i][j] = devarative_function_h(deltha_h[i][1],input[i][2]);
                        devarative_h[i][1] = TranningData[i][j];
                    break;
                case 18 : /// Delta Hidden Layer 2
                        #pragma omp critical
                        TranningData[i][j] = compute_deltha_h(deltha_o[i][1],deltha_o[i][2],weightTebakan[10],weightTebakan[11],output_h[i][2] );
                        deltha_h[i][2] = TranningData[i][j];
                    break;
                case 19 : /// Compute Devarative function sigmoit from hidden layer 2
                        #pragma omp critical
                        TranningData[i][j] = devarative_function_h(deltha_h[i][2],input[i][3]);
                        devarative_h[i][2] = TranningData[i][j];
                    break;
            }

        }
        printf("Data ke - %i | %5.10lf | %5.10lf | %5.10lf | %5.10lf | %5.10lf | %5.10lf | %5.10lf | %5.10lf |\n",i,TranningData[i][12],TranningData[i][13],TranningData[i][14],TranningData[i][15],TranningData[i][16],TranningData[i][17],TranningData[i][18],TranningData[i][19]);
    }
    }

    double stop2 = omp_get_wtime() - start2;
    printf("----------------------------------\n");
    printf("Waktu Eksekusi File    : %lf \n",stop2);
    printf("----------------------------------\n");
    //getchar();
}


void update_weigth(double alpha){

    int i,j;

    printf("----------------------------------------------------------------------------------------------------------------------------------------------------------------\n");
    printf(" %-11s | %-12s | %-12s | %-12s | %-12s | %-12s | %-12s | %-12s | %-12s | %-12s | %-12s | %-12s |","W1","W2","W3","W4","W5","W6","W7","W8","W9","W10","W11","W12"); printf("\n");
    printf("----------------------------------------------------------------------------------------------------------------------------------------------------------------\n");


    for(i=1; i<=nN; i++){
        for(j=20; j<=31; j++){
            switch(j){
                case 20 : ///Update Weight for Output Layer
                        if(i==1){
                                TranningData[i][j] = update_weigth_function(weightTebakan[7],alpha,devarative_o[i][1]);
                                weight[i][7] = TranningData[i][j];
                        }else if(i!=1){
                                TranningData[i][j] = update_weigth_function(weight[i-1][7],alpha,devarative_o[i][1]);
                                weight[i][7] = TranningData[i][j];
                        }
                    break;
                case 21 :
                       if(i==1){
                                TranningData[i][j] = update_weigth_function(weightTebakan[8],alpha,devarative_o[i][1]);
                                weight[i][8] = TranningData[i][j];
                        }else if(i!=1){
                                TranningData[i][j] = update_weigth_function(weight[i-1][8],alpha,devarative_o[i][1]);
                                weight[i][8] = TranningData[i][j];
                        }
                    break;
                case 22 :
                       if(i==1){
                                TranningData[i][j] = update_weigth_function(weightTebakan[9],alpha,devarative_o[i][1]);
                                weight[i][9] = TranningData[i][j];
                        }else if(i!=1){
                                TranningData[i][j] = update_weigth_function(weight[i-1][9],alpha,devarative_o[i][1]);
                                weight[i][9] = TranningData[i][j];
                        }
                    break;



                 case 23 : ///Update Weight for Output Layer
                        if(i==1){
                                TranningData[i][j] = update_weigth_function(weightTebakan[10],alpha,devarative_o[i][2]);
                                weight[i][10] = TranningData[i][j];
                        }else if(i!=1){
                                TranningData[i][j] = update_weigth_function(weight[i-1][10],alpha,devarative_o[i][2]);
                                weight[i][10] = TranningData[i][j];
                        }
                    break;
                case 24 :
                       if(i==1){
                                TranningData[i][j] = update_weigth_function(weightTebakan[11],alpha,devarative_o[i][2]);
                                weight[i][11] = TranningData[i][j];
                        }else if(i!=1){
                                TranningData[i][j] = update_weigth_function(weight[i-1][11],alpha,devarative_o[i][2]);
                                weight[i][11] = TranningData[i][j];
                        }
                    break;
                case 25 :
                       if(i==1){
                                TranningData[i][j] = update_weigth_function(weightTebakan[12],alpha,devarative_o[i][2]);
                                weight[i][12] = TranningData[i][j];
                        }else if(i!=1){
                                TranningData[i][j] = update_weigth_function(weight[i-1][12],alpha,devarative_o[i][2]);
                                weight[i][12] = TranningData[i][j];
                        }
                    break;


                case 26 : ///Update Weight for Output Layer
                        if(i==1){
                                TranningData[i][j] = update_weigth_function(weightTebakan[1],alpha,devarative_h[i][1]);
                                weight[i][1] = TranningData[i][j];
                        }else if(i!=1){
                                TranningData[i][j] = update_weigth_function(weight[i-1][1],alpha,devarative_h[i][1]);
                                weight[i][1] = TranningData[i][j];
                        }
                    break;
                case 27 :
                       if(i==1){
                                TranningData[i][j] = update_weigth_function(weightTebakan[2],alpha,devarative_h[i][1]);
                                weight[i][2] = TranningData[i][j];
                        }else if(i!=1){
                                TranningData[i][j] = update_weigth_function(weight[i-1][2],alpha,devarative_h[i][1]);
                                weight[i][2] = TranningData[i][j];
                        }
                    break;
                case 28 :
                       if(i==1){
                                TranningData[i][j] = update_weigth_function(weightTebakan[3],alpha,devarative_h[i][1]);
                                weight[i][3] = TranningData[i][j];
                        }else if(i!=1){
                                TranningData[i][j] = update_weigth_function(weight[i-1][3],alpha,devarative_h[i][1]);
                                weight[i][3] = TranningData[i][j];
                        }
                    break;




              case 29 : ///Update Weight for Output Layer
                        if(i==1){
                                TranningData[i][j] = update_weigth_function(weightTebakan[4],alpha,devarative_h[i][2]);
                                weight[i][4] = TranningData[i][j];
                        }else if(i!=1){
                                TranningData[i][j] = update_weigth_function(weight[i-1][4],alpha,devarative_h[i][2]);
                                weight[i][4] = TranningData[i][j];
                        }
                    break;
                case 30 :
                       if(i==1){
                                TranningData[i][j] = update_weigth_function(weightTebakan[5],alpha,devarative_h[i][2]);
                                weight[i][5] = TranningData[i][j];
                        }else if(i!=1){
                                TranningData[i][j] = update_weigth_function(weight[i-1][5],alpha,devarative_h[i][2]);
                                weight[i][5] = TranningData[i][j];
                        }
                    break;
                case 31 :
                       if(i==1){
                                TranningData[i][j] = update_weigth_function(weightTebakan[6],alpha,devarative_h[i][2]);
                                weight[i][6] = TranningData[i][j];
                        }else if(i!=1){
                                TranningData[i][j] = update_weigth_function(weight[i-1][6],alpha,devarative_h[i][2]);
                                weight[i][6] = TranningData[i][j];
                        }
                    break;
                }

            }
            printf(" %i | %5.5lf | %5.5lf | %5.5lf | %5.5lf | %5.5lf | %5.5lf | %5.5lf | %5.5lf | %5.5lf | %5.5lf | %5.5lf | %5.5lf | \n",i,TranningData[i][26],TranningData[i][27],TranningData[i][28],TranningData[i][29],TranningData[i][30],TranningData[i][31],TranningData[i][20],TranningData[i][21],TranningData[i][22],TranningData[i][23],TranningData[i][24],TranningData[i][25]);
        }

        /// Update Weight
        /// Weight for Hidden Layer
        int index;
        for(index = 1 ; index<=12; index++){
                weightTebakan[index]= weight[i-1][index];
                printf("W%i :%lf \n",index,weightTebakan[index]);
        }
    }

int main () {
  n=150;
    N=1;
    nN=1500;
    double alpha =0.00001;
    double w1, w2, w3;
    double mse = 0;
    /// Weight for Hidden Layer
    weightTebakan[1]= 0.15;
    weightTebakan[2]= 0.2;
    weightTebakan[3]= 0.35;

    /// Weight for Hidden Layer 2
    weightTebakan[4]= 0.25;
    weightTebakan[5]= 0.3;
    weightTebakan[6]= 0.35;

    /// Weight for Output Layer
    weightTebakan[7]= 0.4;
    weightTebakan[8]= 0.45;
    weightTebakan[9]= 0.6;

    /// Weight for Output Layer 2
    weightTebakan[10]= 0.5;
    weightTebakan[11]= 0.55;
    weightTebakan[12]= 0.6;




    getDataTrainning_fromFile("JSTFILE.txt");
    double start3 = omp_get_wtime();
    int itr = 1;
    while ( itr<=1){
        Forward_Pass();
        Backward_Pass();
        update_weigth(alpha);
        mse=mean_se();
        printf("\n MSE : %lf \n",mean_se());
        itr+=1;
    }
    double stop3 = omp_get_wtime() - start3;
    printf("----------------------------------\n");
    printf("Waktu Eksekusi File    : %lf \n",stop3);
    printf("----------------------------------\n");
    getchar();

    return 0;
}


void getDataTrainning_fromFile(char* file) /// Fungsi mengambil data trainning dari file
{
    FILE *arr; double inputF[1501][20]; int x,y;  /// inputF => input Form FIle

    arr =fopen(file,"r");
    printf("\n\n\n\n");
    printf("\t\t\t\t\t\t ");
    printf("Data Trainning \n");
    printf("\t\t\t\t\t\t ");
    printf("----------------------------------\n");
    printf("\t\t\t\t\t\t ");
    printf(" %-2s | %-4s | %-4s | %-4s | \n","No","X1","X2","Class");
    printf("\t\t\t\t\t\t ");
    printf("----------------------------------\n");
    for(x=1;x<=nN;x++){
        printf("\t\t\t\t\t\t ");
        for(y=1;y<=4;y++){
            fscanf(arr, "%lf ",&inputF[x][y]);
            input[x][y]=inputF[x][y];

            /// Print out Data Trainning
            if(y==1 || y==4){
                      printf("%5.2lf | ",  input[x][y]);
                }else{
                     if(input[x][y]>=10){
                         printf("%5.2lf | ",  input[x][y]);
                    }else if(  input[x][y]>=0 &&   input[x][y]<=10 ) {
                         printf("%5.2lf | ",  input[x][y]);
                    }else if(  input[x][y]>=-10 &&   input[x][y]<=0) {
                         printf("%5.2lf | ",  input[x][y]);
                    }else if(  input[x][y]<=-10){
                         printf("%5.2lf | ",  input[x][y]);
                    }
                }
        }printf("\n");
    }
    getchar();
    printf("\t\t\t\t\t\t ");
    printf("----------------------------------\n");

}



