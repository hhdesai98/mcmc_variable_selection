/*
 This program computes the log marginal likelihoods of of several regressions in parallel.
 Compile the program using the makefile provided.
 
 Run the program using the command:

 mpirun -np 10 parallelR2 
*/

#include <stdio.h>
#include <stdlib.h>
#include <float.h>
#include <math.h>
#include <mpi.h>
#include <iomanip>
#include <gsl/gsl_math.h>
#include <gsl/gsl_rng.h>
#include <gsl/gsl_vector.h>
#include <gsl/gsl_matrix.h>
#include <gsl/gsl_statistics.h>
#include <gsl/gsl_randist.h>
#include <gsl/gsl_cdf.h>
#include <gsl/gsl_blas.h>
#include <gsl/gsl_permutation.h>
#include <gsl/gsl_linalg.h>
#include <gsl/gsl_sort_double.h>
#include <gsl/gsl_sort_vector.h>
#include <gsl/gsl_errno.h>
#include <gsl/gsl_eigen.h>

// For MPI communication
#define BAYESLOGISTIC	1
#define SHUTDOWNTAG	0

// Used to determine PRIMARY or REPLICA
static int myrank;

// Global variables
int numsamps = 10000;
int n = 148; 
int p = 61;
int nvariables = 60;
gsl_matrix* X = gsl_matrix_alloc(n,p-1);
gsl_matrix* y = gsl_matrix_alloc(n,1);

gsl_rng* mystream;

// Function Declarations
void primary();
void replica(int primaryname);

//declare regression list structure
typedef struct myRegression* LPRegression;
typedef struct myRegression Regression;

struct myRegression
{
  int A; //regressor 
  double logmarglikA; //laplace est of log marginal likelihood of the regression
  double MC; //MC estimate
  double beta_0_post;
  double beta_1_post;
  LPRegression Next; //link to the next regression
};
void DeleteLastRegression(LPRegression regressions);
void SaveRegressions(char* filename,LPRegression regressions);
void AddRegression2(int nMaxRegs, LPRegression regressions,int A,double logmarglikA, double MC,double beta_0_post,double beta_1_post);

//define auxilary functions from R hw#4
double vector_sum(gsl_vector* v);
double logdet(gsl_matrix* m);
double invlogit(double x);
gsl_matrix* invlogit_mat(gsl_matrix* x);
double invlogit2(double x);
gsl_matrix* invlogit2_mat(gsl_matrix* x);
void matrixproduct(gsl_matrix* m1,gsl_matrix* m2,gsl_matrix* m);
gsl_matrix* getPi(gsl_matrix* x, gsl_matrix* beta);
gsl_matrix* getPi(gsl_matrix* x, gsl_matrix* beta);
double logisticLogLik(gsl_matrix* y, gsl_matrix* x, gsl_matrix* beta);
double lstar(gsl_matrix* y, gsl_matrix* x, gsl_matrix* beta);
void getGradient(gsl_matrix* y, gsl_matrix* x, gsl_matrix* beta, gsl_matrix* gradient);
void getHessian(gsl_matrix* y, gsl_matrix* x, gsl_matrix* beta, gsl_matrix*hessian);
void invert(gsl_matrix* m, gsl_matrix* inverse);
gsl_matrix* getcoefNR(gsl_matrix* y, gsl_matrix* x);
void mhLogisticRegression(gsl_rng* mystream, gsl_matrix* y, gsl_matrix* x, gsl_matrix* beta, gsl_matrix* invNegHessian, gsl_matrix* betaCurrent);
double getLaplaceApprox(gsl_matrix* y, gsl_matrix* x, gsl_matrix* betaMode);
double getMonteCarlo(gsl_rng* mystream, gsl_matrix* y, gsl_matrix* x, int NumberOfIterations);
gsl_matrix* getPosteriorMeans(gsl_rng* mystream, gsl_matrix* y, gsl_matrix* x, gsl_matrix* betaMode, int NumberOfIterations);
void subset_gsl_matrix(gsl_matrix* fulldata, gsl_matrix* subdata, int n, int* A, int lenA);
gsl_matrix* getXcol(gsl_matrix*X, int j);
void bayesLogistic(int j,double workresults[5]);

int main(int argc, char* argv[])
{

   ///////////////////////////
   // START THE MPI SESSION //
   ///////////////////////////
   MPI_Init(&argc, &argv);

   /////////////////////////////////////
   // What is the ID for the process? //   
   /////////////////////////////////////
   MPI_Comm_rank(MPI_COMM_WORLD,&myrank);

   // Read in the data
   //load the data
   
   char datafilename[] = "534finalprojectdata.txt";

	gsl_matrix* data = gsl_matrix_alloc(n,p);

	FILE* datafile = fopen(datafilename,"r");
	if(NULL==datafile)
	{
		fprintf(stderr,"Cannot open data file [%s]\n",datafilename);
		return(0);
	}
	if(0!=gsl_matrix_fscanf(datafile,data))
	{
		fprintf(stderr,"File [%s] does not have the required format.\n",datafilename);
		return(0);
	}
	fclose(datafile);

   int* y_index = new int[1];
   y_index[0] = 60;
   subset_gsl_matrix(data, y, n, y_index, 1);

   int* x_index = new int[p-1];
   for(int i=0; i<p-1;i++)
   {
      x_index[i] = i;
   }

   subset_gsl_matrix(data, X, n, x_index, p - 1);

   //intialize random stream
   const gsl_rng_type* T;

   gsl_rng_env_setup();

   T = gsl_rng_default;
   mystream = gsl_rng_alloc(T);


   // Branch off to primary or replica function
   // Primary has ID == 0, the replicas are then in order 1,2,3,...

   if(myrank==0)
   {
      primary();
   }
   else
   {
      replica(myrank);
   }

   gsl_matrix_free(X);
   gsl_matrix_free(y);
   gsl_rng_free(mystream);
   gsl_matrix_free(data);
   delete[] x_index;
   delete[] y_index;

   // Finalize the MPI session
   MPI_Finalize();

   return(1);
}

void primary()
{
   int var;		// to loop over the variables
   int rank;		// another looping variable
   int ntasks;		// the total number of replicas
   int jobsRunning;	// how many replicas we have working
   int work[1];		// information to send to the replicas
   double workresults[5]; // info received from the replicas
   MPI_Status status;	// MPI information
   char outputfilename[] = "regressions.txt";

   // Find out how many replicas there are
   MPI_Comm_size(MPI_COMM_WORLD,&ntasks);

   fprintf(stdout, "Total Number of processors = %d\n",ntasks);

   // Now loop through the variables and compute the R2 values in
   // parallel
   jobsRunning = 1;

   //create the head of the list of regressions
   LPRegression regressions = new Regression;
   //properly mark the end of the list
   regressions->Next = NULL;

   for(var=0; var<nvariables; var++)
   {
      // This will tell a replica which variable to work on
      work[0] = var;

      if(jobsRunning < ntasks) // Do we have an available processor?
      {
         // Send out a work request
         MPI_Send(&work, 	// the vector with the variable
		            1, 		// the size of the vector
		            MPI_INT,	// the type of the vector
                  jobsRunning,	// the ID of the replica to use
                  BAYESLOGISTIC,	// tells the replica what to do
                  MPI_COMM_WORLD); // send the request out to anyone
				   // who is available
         printf("Primary sends out work request [%d] to replica [%d]\n",
                work[0],jobsRunning);

         // Increase the # of processors in use
         jobsRunning++;

      }
      else // all the processors are in use!
      {
         MPI_Recv(workresults,	// where to store the results
 		            5,		// the size of the vector
		            MPI_DOUBLE,	// the type of the vector
	 	            MPI_ANY_SOURCE,
		            MPI_ANY_TAG, 	
		            MPI_COMM_WORLD,
		            &status);     // lets us know which processor
				// returned these results

         printf("Primary has received the result of work request [%d] from replica [%d]\n",
                 (int) workresults[0],status.MPI_SOURCE);
 
         AddRegression2(5, regressions,(int)workresults[0]+1,workresults[1],workresults[2], workresults[3], workresults[4]);


         printf("Primary sends out work request [%d] to replica [%d]\n",
                work[0],status.MPI_SOURCE);

         // Send out a new work order to the processors that just
         // returned
         MPI_Send(&work,
                  1,
                  MPI_INT,
                  status.MPI_SOURCE, // the replica that just returned
                  BAYESLOGISTIC,
                  MPI_COMM_WORLD); 
      } // using all the processors
   } // loop over all the variables


   ///////////////////////////////////////////////////////////////
   // NOTE: we still have some work requests out that need to be
   // collected. Collect those results now.
   ///////////////////////////////////////////////////////////////

   // loop over all the replicas
   for(rank=1; rank<jobsRunning; rank++)
   {
      MPI_Recv(workresults,
               5,
               MPI_DOUBLE,
               MPI_ANY_SOURCE,	// whoever is ready to report back
               MPI_ANY_TAG,
               MPI_COMM_WORLD,
               &status);

       printf("Primary has received the result of work request [%d]\n",
                (int)workresults[0]);
 
      //save the results received
      AddRegression2(5, regressions,(int)workresults[0]+1,workresults[1],workresults[2], workresults[3], workresults[4]);
      
   }

   printf("Tell the replicas to shutdown.\n");

   // Shut down the replica processes
   for(rank=1; rank<ntasks; rank++)
   {
      printf("Primary is shutting down replica [%d]\n",rank);
      MPI_Send(0,
	            0,
               MPI_INT,
               rank,		// shutdown this particular node
               SHUTDOWNTAG,		// tell it to shutdown
	       MPI_COMM_WORLD);
   }

   printf("got to the end of Primary code\n");

   SaveRegressions(outputfilename,regressions);
   

   // return to the main function
   return;
  
}

void replica(int replicaname)
{
   int work[1];			// the input from primary
   double workresults[5];	// the output for primary
   MPI_Status status;		// for MPI communication
   // the replica listens for instructions...
   int notDone = 1;
   while(notDone)
   {
      printf("Replica %d is waiting\n",replicaname);
      MPI_Recv(&work, // the input from primary
	            1,		// the size of the input
	            MPI_INT,		// the type of the input
               0,		// from the PRIMARY node (rank=0)
               MPI_ANY_TAG,	// any type of order is fine
               MPI_COMM_WORLD,
               &status);
      printf("Replica %d just received smth\n",replicaname);

      // switch on the type of work request
      switch(status.MPI_TAG)
      {
         case BAYESLOGISTIC:
            // Get the logistic values for this variable
            // ...and save it in the results vector

           printf("Replica %d has received work request [%d]\n",
                  replicaname,work[0]);
          
            
            // tell the primary what variable you're returning
            bayesLogistic(work[0], workresults);

            // Send the results
            MPI_Send(&workresults,
                     5,
                     MPI_DOUBLE,
                     0,		// send it to primary
                     0,		// doesn't need a TAG
                     MPI_COMM_WORLD);

            printf("Replica %d finished processing work request [%d]\n",
                   replicaname,work[0]);

            break;

         case SHUTDOWNTAG:
         {
            printf("Replica %d was told to shutdown\n",replicaname);
            return;
         }
         default:
         {
            notDone = 0;
            printf("The replica code should never get here.\n");
            return;
         }
      }
   }
   return;
}

//this function deletes the last element of the list
//with the head "regressions"
//again, the head is not touched
void DeleteLastRegression(LPRegression regressions)
{
  //this is the element before the first regression
  LPRegression pprev = regressions;
  //this is the first regression
  LPRegression p = regressions->Next;

  //if the list does not have any elements, return
  if(NULL==p)
  {
     return;
  }

  //the last element of the list is the only
  //element that has the "Next" field equal to NULL
  while(NULL!=p->Next)
  {
    pprev = p;
    p = p->Next;
  }
  
  //now "p" should give the last element
  //delete it

  p->Next = NULL;
  delete p;

  //now the previous element in the list
  //becomes the last element
  pprev->Next = NULL;

  return;
}

// this function checks whether our new regression is in the top "nMaxRegs"
// based on marginal likelihood. If the new model is better than any of the
// existing regressions in the list, it is added to the list
// and the worst regression is discarded. Here "regressions" represents
// the head of the list, "lenA" is the number of predictors
// and "logmarglikA" is the marginal likelihood of the regression
// with predictors A.
void AddRegression2(int nMaxRegs, LPRegression regressions,int A,double logmarglikA, double MC,double beta_0_post,double beta_1_post)
{
  int i, j = 0;

  LPRegression p = regressions;
  LPRegression pnext = p->Next;

  while(NULL != pnext && j < nMaxRegs)
  {
     

     //go to the next element in the list if the current
     //regression has a larger log marginal likelihood than
     //the new regression A
     if(pnext->MC > MC)
     {
        p = pnext;
        pnext = p->Next;
     }
     else //otherwise stop; this is where we insert the new regression
     {
        break;
     }
     j++;
  }

  // if we reached "nMaxRegs" we did not beat any of the top 10 with the new regression.
  // Otherwise we add it like normal.

  if(nMaxRegs == j)
  {
	  return;
  }

  //create a new element of the list
  LPRegression newp = new Regression;
  newp->A = A;
  newp->logmarglikA = logmarglikA;
  newp->MC = MC;
  newp->beta_0_post = beta_0_post;
  newp->beta_1_post = beta_1_post;
  

  //insert the new element in the list
  p->Next = newp;
  newp->Next = pnext;

  // now we move through the list until we either reach the end of it, or reach the
  // element just after the "nMaxRegs" element.
  while(j < nMaxRegs && NULL!=pnext)
  {
	  p = pnext;
	  pnext = p->Next;
	  j++;
  }
  // if we reach nMaxRegs, we have to discard the new worst element in the list.
  if(nMaxRegs == j)
  {
	  DeleteLastRegression(regressions);
  }

  return;
}

//this function saves the regressions in the list with
//head "regressions" in a file with name "filename"
void SaveRegressions(char* filename,LPRegression regressions)
{  

  int i;
  //open the output file
  FILE* out = fopen(filename,"w");
	
  if(NULL==out)
  {
    printf("Cannot open output file [%s]\n",filename);
    exit(1);
  }

  //this is the first regression
  LPRegression p = regressions->Next;
  fprintf(out,"Variable Laplace        MC_Approx       beta_0          beta_1\n");
  while(NULL!=p)
  {
    //print the log marginal likelhood and the number of predictors
    fprintf(out,"%d\t%.5lf\t%.5lf\t%.5lf\t%.5lf\n",p->A,p->logmarglikA,p->MC,p->beta_0_post,p->beta_1_post);

    //go to the next regression
    p = p->Next;
  }

  //close the output file
  fclose(out);

  return;
}

void subset_gsl_matrix(gsl_matrix* fulldata, gsl_matrix* subdata, int n, int* A, int lenA)
{
	gsl_vector* tempvec = gsl_vector_alloc(n);
	int i;

	//set the columns of the matrix.
	for(i = 0; i < lenA; i++)
	{
		gsl_matrix_get_col(tempvec, fulldata, (A[i]));
		gsl_matrix_set_col(subdata, i, tempvec);
	}
	// clean the memory
	gsl_vector_free(tempvec);
	return;
}

//calculates the product of a nxp matrix m1 with a pxl matrix m2
//returns a nxl matrix m
void matrixproduct(gsl_matrix* m1,gsl_matrix* m2,gsl_matrix* m)
{
	int i,j,k;
	double s;
	
	for(i=0;i<m->size1;i++)
	{
	  for(k=0;k<m->size2;k++)
	  {
	    s = 0;
	    for(j=0;j<m1->size2;j++)
	    {
	      s += gsl_matrix_get(m1,i,j)*gsl_matrix_get(m2,j,k);
	    }
	    gsl_matrix_set(m,i,k,s);
	  }
	}
	return;
}

//this function inverts matrices using the LU decomposition
void invert(gsl_matrix* m, gsl_matrix* inverse)
{
   gsl_permutation* p = gsl_permutation_alloc(m->size1);

   int s;

   gsl_linalg_LU_decomp(m,p,&s);
   
   gsl_linalg_LU_invert(m,p,inverse);

   gsl_permutation_free(p);

   return;
}

//this function takes in the data matrix X, and the column index to extract (0 indexed)
// it returns a nx2 matrix where the first column is the column of 1's
gsl_matrix* getXcol(gsl_matrix*X, int j)
{
   gsl_matrix* x_col = gsl_matrix_alloc(X->size1,2);
   for(int i=0;i<X->size1;i++)
   {
      gsl_matrix_set(x_col, i, 0, 1);
      gsl_matrix_set(x_col,i,1,gsl_matrix_get(X,i,j));
   }
   return(x_col);
}

//this function computes the sum of a vectors elements
double vector_sum(gsl_vector* v)
{
   double sum = 0;
   for (int i=0; i<v->size;i++)
   {
      sum+= gsl_vector_get(v,i);
   }
   return(sum);
}

//this function computes the logdet of a matrix
double logdet(gsl_matrix* m)
{
   int n = m->size1;
   //intialize eigen symm workspace
   gsl_eigen_symm_workspace *w = gsl_eigen_symm_alloc(n);

   gsl_vector* eigen_vals = gsl_vector_alloc(n);

   //find and store eigenvalues
   gsl_eigen_symm(m,eigen_vals,w);

   gsl_eigen_symm_free(w);

   //take log of each value
   for (int i=0;i<n;i++)
   {
      gsl_vector_set(eigen_vals,i,log(gsl_vector_get(eigen_vals,i)));
   }

   //sum the values
   double log_det = vector_sum(eigen_vals);

   gsl_vector_free(eigen_vals);

   return(log_det);
   
}

//invlogit function
double invlogit(double x)
{
   return(exp(x)/(1+exp(x)));
}

//this function takes the inv_logit transformation of a matrix
gsl_matrix* invlogit_mat(gsl_matrix* x)
{
   int n = x->size1;
   int p = x->size2;
   gsl_matrix* x_inv_logit = gsl_matrix_alloc(n,p);

   for(int i=0;i<n;i++)
   {
      for(int j=0;j<p;j++)
      {
         gsl_matrix_set(x_inv_logit,i,j,invlogit(gsl_matrix_get(x,i,j)));
      }
   }
   return(x_inv_logit);
}

//this function does the invlogit2 transform
double invlogit2(double x)
{
   return(exp(x)/pow(1+exp(x),2));
}

//applies invlogit2 to each element of the matrix x
gsl_matrix* invlogit2_mat(gsl_matrix* x)
{
   int n = x->size1;
   int p = x->size2;
   gsl_matrix* x_inv_logit = gsl_matrix_alloc(n,p);

   for(int i=0;i<n;i++)
   {
      for(int j=0;j<p;j++)
      {
         gsl_matrix_set(x_inv_logit,i,j,invlogit2(gsl_matrix_get(x,i,j)));
      }
   }
   return(x_inv_logit);
}

//this function calculates invlogit(XB), X is needed as a 2x1 matrix with the 1's column
//this is used in the gradient
gsl_matrix* getPi(gsl_matrix* x, gsl_matrix* beta)
{
   int n = x->size1;
   gsl_matrix* xb = gsl_matrix_alloc(n,1);
   matrixproduct(x,beta,xb);
   gsl_matrix* xb_inv = invlogit_mat(xb);

   gsl_matrix_free(xb);

   return(xb_inv);
}

//this functions calculates invlogit2(XB)
//this is used in the hessian
gsl_matrix* getPi2(gsl_matrix* x, gsl_matrix* beta)
{
   int n = x->size1;
   gsl_matrix* xb = gsl_matrix_alloc(n,1);
   matrixproduct(x,beta,xb);
   gsl_matrix* xb_inv2 = invlogit2_mat(xb);

   gsl_matrix_free(xb);

   return(xb_inv2);
}

//this function finds the logistic log likelihood of a regression
double logisticLogLik(gsl_matrix* y, gsl_matrix* x, gsl_matrix* beta)
{
   //intialize values
   int n = y->size1;
   gsl_matrix* pi = getPi(x,beta);

   double y_log_pi = 0;
   double y_1_log_pi = 0;

   for (int i=0;i<n;i++)
   {
      //add y*log(pi)
      y_log_pi += gsl_matrix_get(y,i,0)*log(gsl_matrix_get(pi,i,0));
      //add (1-y)*log(1-pi)
      y_1_log_pi += (1-gsl_matrix_get(y,i,0))*log(1-gsl_matrix_get(pi,i,0));
      
   }

   double log_lik = y_log_pi + y_1_log_pi;
   //free matrices
   gsl_matrix_free(pi);
   return(log_lik);


}

//this finds l^star of a specific beta
double lstar(gsl_matrix* y, gsl_matrix* x, gsl_matrix* beta)
{
   double mean_beta = 0;
   for (int i=0;i<beta->size1;i++){
      mean_beta += pow(gsl_matrix_get(beta,i,0),2);
   }
   mean_beta = mean_beta/2;
   mean_beta = mean_beta*(-1);
   mean_beta = mean_beta + logisticLogLik(y,x,beta);

   return(mean_beta);
}

//this calculates the gradient of the logistic log loss and stores it in a gsl_matrix gradient
void getGradient(gsl_matrix* y, gsl_matrix* x, gsl_matrix* beta, gsl_matrix* gradient)
{
   //get pi
   gsl_matrix* pi = getPi(x,beta);
   //allocate and store y-pi
   gsl_matrix* y_min_pi = gsl_matrix_alloc(y->size1,y->size2);
   gsl_matrix_memcpy(y_min_pi,y);
   gsl_matrix_sub(y_min_pi,pi);

   //sum elements
   double sum_y_pi = 0;
   for(int i=0;i<y->size1;i++)
   {
      sum_y_pi+=gsl_matrix_get(y_min_pi,i,0);
   }

   //store beta_0 gradient
   gsl_matrix_set(gradient,0,0,sum_y_pi - gsl_matrix_get(beta,0,0));
   
   //find sum((y-pi)*x)
   double sum_y_pi_x = 0;

   for(int i=0;i<y->size1;i++)
   {
      sum_y_pi_x+=gsl_matrix_get(y_min_pi,i,0)*gsl_matrix_get(x,i,1);
   }

   //set beta_1 gradient
   gsl_matrix_set(gradient,1,0,sum_y_pi_x - gsl_matrix_get(beta,1,0));

   //free memory
   gsl_matrix_free(y_min_pi);
   gsl_matrix_free(pi);

   return;
}


//this function finds the hessian for the beta values and stores it in a gsl_matrix
void getHessian(gsl_matrix* y, gsl_matrix* x, gsl_matrix* beta, gsl_matrix* hessian)
{
   
   gsl_matrix* pi2 = getPi2(x,beta);

   double hes_0_0 = 1;
   double hes_0_1 = 0;
   double hes_1_1 = 1;
   for(int i = 0; i<pi2->size1; i++)
   {
      //sum(pi2)
      hes_0_0 += gsl_matrix_get(pi2,i,0);
      //sum(pi2*x)
      hes_0_1 += gsl_matrix_get(pi2,i,0)*gsl_matrix_get(x,i,1);
      //sum(pi2*x^2)
      hes_1_1 += gsl_matrix_get(pi2,i,0)*pow(gsl_matrix_get(x,i,1),2);
   }

   gsl_matrix_set(hessian,0,0,hes_0_0);
   gsl_matrix_set(hessian,0,1,hes_0_1);
   gsl_matrix_set(hessian,1,0,hes_0_1);
   gsl_matrix_set(hessian,1,1,hes_1_1);

   gsl_matrix_free(pi2);
   gsl_matrix_scale(hessian,-1);

   return;
}

//This function finds the coefficients of a logistic regression using the Newton Raphson Method
//It takes in y and x, where x is a nx2 matrix with the intercept as the first column
gsl_matrix* getcoefNR(gsl_matrix* y, gsl_matrix* x)
{
   //denote tolerance
   double epsilon = 1e-10;

   //initialize beta matrices
   gsl_matrix* beta = gsl_matrix_alloc(2,1);
   gsl_matrix* new_beta = gsl_matrix_alloc(2,1);
   gsl_matrix_set_all(beta, 0);  

   //find current lstar (of 0 matrix)
   double currentLStar = lstar(y,x,beta);

   double iteration = 0;

   //allocate matrices for the gradient and inverse hessian
   gsl_matrix* hessian = gsl_matrix_alloc(2,2);
   gsl_matrix* gradient = gsl_matrix_alloc(2,1);
   gsl_matrix* inv_hes = gsl_matrix_alloc(2,2);

   while(1)
   {
      //update gradients and hessian
      iteration++;
      getHessian(y,x,beta,hessian);
      getGradient(y,x,beta,gradient);
      invert(hessian, inv_hes);
      
      //find update term, and multiply it by -1
      matrixproduct(inv_hes,gradient,new_beta);
      gsl_matrix_scale(new_beta,-1);

      //beta - new_beta and store it in new_beta
      gsl_matrix_add(new_beta,beta);

      //find new lstar
      double newLstar = lstar(y,x,new_beta);

      //if the tolerance is met break
      if((abs(gsl_matrix_get(new_beta,0,0)-gsl_matrix_get(beta,0,0)) < epsilon) && (abs(gsl_matrix_get(new_beta,1,0)-gsl_matrix_get(beta,1,0)) < epsilon))
      {
         break;
      }

      //update beta as new beta
      gsl_matrix_memcpy(beta,new_beta);
      currentLStar = newLstar;
   }
   //free matrices
   gsl_matrix_free(new_beta);
   gsl_matrix_free(hessian);
   gsl_matrix_free(gradient);
   gsl_matrix_free(inv_hes);

   return(beta);
}

//this function does one step of the metropolis hastings algorithm
//it takes in a random variable stream, y, x, a beta value, the inverse negative hessian, and the a matrix to store the new value in
void mhLogisticRegression(gsl_rng* mystream, gsl_matrix* y, gsl_matrix* x, gsl_matrix* beta, gsl_matrix* invNegHessian,gsl_matrix* betaCurrent)
{
   //initialize candidate vector
   gsl_vector* betaCandidate = gsl_vector_alloc(2);

   //intialize components to sample from the multivariate gaussian distribution
   gsl_vector* mu = gsl_vector_alloc(2);
   gsl_vector_set(mu,0,gsl_matrix_get(beta,0,0));
   gsl_vector_set(mu,1,gsl_matrix_get(beta,1,0));
   gsl_matrix* L = gsl_matrix_alloc(2,2);
   gsl_matrix_memcpy(L,invNegHessian);
   gsl_linalg_cholesky_decomp(L);
   gsl_matrix_set(L,0,1,0);

   //generate random value and store it in betaCandidate
   gsl_ran_multivariate_gaussian(mystream, mu, L, betaCandidate);

   //create a matrix of betaCandidate
   gsl_matrix* betaCanMat = gsl_matrix_alloc(2,1);
   gsl_matrix_set(betaCanMat, 0, 0, gsl_vector_get(betaCandidate,0));
   gsl_matrix_set(betaCanMat, 1, 0, gsl_vector_get(betaCandidate,1));

   //store metrics
   double currentLStar = lstar(y,x,beta);

   double candidateLStar = lstar(y,x,betaCanMat);

   //if the candidate has a better l_star, update the matrix betacurrent with the candidate
   if(candidateLStar>=currentLStar)
   {
      gsl_matrix_memcpy(betaCurrent,betaCanMat); 
   }

   //else randomly samply from (0,1)
   double u = gsl_ran_flat(mystream, 0, 1);

   if(u<=exp(candidateLStar-currentLStar))
   {
      //accept the move and copy the matrix
      gsl_matrix_memcpy(betaCurrent,betaCanMat); 
   } 
   else
   {
      //reject the move and stay at the current state
      gsl_matrix_memcpy(betaCurrent,beta); 
   }

  gsl_vector_free(betaCandidate);
  gsl_matrix_free(betaCanMat);
  gsl_vector_free(mu);
  gsl_matrix_free(L);

  return;
}

double getLaplaceApprox(gsl_matrix* y, gsl_matrix* x, gsl_matrix* betaMode)
{
   //find logistic log likelihood with betaMode
   double maxLogLik = logisticLogLik(y,x,betaMode);

   //find -mean(beta^2)
   double mean_beta_mode_2 = 0;
   for (int i=0;i<betaMode->size1;i++)
   {
      mean_beta_mode_2 += -pow(gsl_matrix_get(betaMode,i,0),2)/2;
   }

   //find negative hessian
   gsl_matrix* betaMode_hes = gsl_matrix_alloc(2,2);
   getHessian(y,x,betaMode, betaMode_hes);
   gsl_matrix_scale(betaMode_hes,-1);

   //calculate laplace approx to logmarglik
   double logmarglik = mean_beta_mode_2+maxLogLik-0.5*logdet(betaMode_hes);

   gsl_matrix_free(betaMode_hes);

   return(logmarglik);

}

//this function computes the monte carlo integration to find the MC log likelihood constant
//it takes in a stream, y, x and a number of iterations
double getMonteCarlo(gsl_rng* mystream, gsl_matrix* y, gsl_matrix* x, int NumberOfIterations)
{
   //intialize vector to store log likelihoods
   gsl_vector* loglikVec = gsl_vector_alloc(NumberOfIterations);

   //initialize mu vector
   gsl_vector* mu = gsl_vector_alloc(2);
   gsl_vector_set_zero(mu);

   //initalize covariance matrix
   gsl_matrix* L = gsl_matrix_alloc(2,2);
   gsl_matrix_set_identity(L);
   gsl_linalg_cholesky_decomp(L);
   gsl_matrix_set(L,0,1,0);

   //intialize sampled vectors
   gsl_vector* ran_beta = gsl_vector_alloc(2);
   gsl_matrix* ran_beta_mat = gsl_matrix_alloc(2,1);

   for(int i=0;i<NumberOfIterations;i++)
   {
      //sample a candidate
      gsl_ran_multivariate_gaussian(mystream, mu, L, ran_beta);
      //set the ran_beta equal to a matrix to then use logisticLogLike
      gsl_matrix_set(ran_beta_mat,0,0,gsl_vector_get(ran_beta,0));
      gsl_matrix_set(ran_beta_mat,1,0,gsl_vector_get(ran_beta,1));
      //set the logisticLogLik for that sampled vector
      gsl_vector_set(loglikVec,i,logisticLogLik(y,x,ran_beta_mat));
   }

   //extract the max of the log likelihoods
   double maxloglikVec = gsl_vector_max(loglikVec);

   //take exp(loglik-maxloglik)
   gsl_vector* exp_log_lik = gsl_vector_alloc(NumberOfIterations);

   for (int i=0;i<NumberOfIterations;i++)
   {
      gsl_vector_set(exp_log_lik,i, exp(gsl_vector_get(loglikVec,i)-maxloglikVec));
   }

   //log(mean(loglikvec))
   double log_mean_lik = vector_sum(exp_log_lik);
   log_mean_lik = log(log_mean_lik/(double)NumberOfIterations);

   gsl_vector_free(loglikVec);
   gsl_matrix_free(ran_beta_mat);
   gsl_vector_free(ran_beta);
   gsl_vector_free(exp_log_lik);
   gsl_matrix_free(L);

   //return scaled sum
   return(log_mean_lik + maxloglikVec);
}

//this function finds the posterior means of beta_0 and beta_1
//it takes in a stream, y, x, betaMode (from NR), and number of iterations
gsl_matrix* getPosteriorMeans(gsl_rng* mystream, gsl_matrix* y, gsl_matrix* x, gsl_matrix* betaMode, int NumberOfIterations)
{
   //first we intialize betaBayes as 0
   gsl_matrix* betaBayes = gsl_matrix_alloc(2,1);
   gsl_matrix_set_zero(betaBayes);

   //we find the negative inv_hessian at betaMode
   gsl_matrix* hes = gsl_matrix_alloc(2,2);
   getHessian(y,x,betaMode, hes);
   gsl_matrix* inv_hes = gsl_matrix_alloc(2,2);
   invert(hes, inv_hes);
   gsl_matrix_scale(inv_hes,-1);

   //we intialize values for beta old and current
   gsl_matrix* betaCurrent = gsl_matrix_alloc(2,1);
   gsl_matrix* betaold = gsl_matrix_alloc(2,1);
   gsl_matrix_memcpy(betaCurrent,betaMode);
   gsl_matrix_memcpy(betaold,betaMode);

   for(int i = 0; i<NumberOfIterations; i++)
   {  
      //we sample a value and store it in betacurrent, with the previous sample as the mean
      mhLogisticRegression(mystream, y,x,betaold, inv_hes, betaCurrent);
      //we add the two 2x1 matrices
      gsl_matrix_add(betaBayes,betaCurrent);
      //we update betaold with the new sample
      gsl_matrix_memcpy(betaold,betaCurrent);
   }

   //we scale betaBayes by the number of iterations
   double scale_fac = pow(NumberOfIterations,-1);
   gsl_matrix_scale(betaBayes,scale_fac);
   gsl_matrix_free(betaCurrent);
   gsl_matrix_free(inv_hes);
   gsl_matrix_free(hes);

   return(betaBayes);
}

//this function takes in a (0-indexed) column index and finds the LaplaceApprox, the MC Approx, and posterior means with the logistic regression associated with that variable
//it stores these results in workresults
void bayesLogistic(int j, double workresults[5])
{
   gsl_matrix* x = getXcol(X,j);
   gsl_matrix* betaMode = getcoefNR(y,x);
   
   double logmarglik  = getLaplaceApprox(y,x,betaMode);

   double logmarglikMC = getMonteCarlo(mystream,y,x,numsamps);

   gsl_matrix* betaBayes = getPosteriorMeans(mystream, y, x, betaMode,numsamps);

   workresults[0] = j;
   workresults[1] = logmarglik;
   workresults[2] = logmarglikMC;
   workresults[3] = gsl_matrix_get(betaBayes,0,0);
   workresults[4] = gsl_matrix_get(betaBayes,1,0);

   gsl_matrix_free(betaBayes);
   gsl_matrix_free(betaMode);
   gsl_matrix_free(x);

   return;
}