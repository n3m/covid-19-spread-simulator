#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <cstring>
#include <ctime>
#include <math.h>
#include <time.h>
#include <random>
#include <chrono>
#include <vector>
#include <string>
#include <sstream>

#include "classes.cpp"

/* CUDA Runtime */
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <curand_kernel.h>

__host__ void check_CUDA_error(const char* msg) {
	cudaError_t err;
	cudaDeviceSynchronize();
	err = cudaGetLastError();
	if (err != cudaSuccess) {
		printf("Error (%d): [%s] %s\n", err, msg, cudaGetErrorString(err));
	}
}

using namespace std;

class Agent;
class Operations;
class GlobalState;

typedef std::chrono::high_resolution_clock myclock;
myclock::time_point beginning = myclock::now();
myclock::duration d = myclock::now() - beginning;
unsigned seed2 = d.count();
std::mt19937 rng(seed2);
std::default_random_engine generator;

__global__ void randomCudaGenerator(int min, int max, double* result)
{
	int tId = threadIdx.x + (blockIdx.x * blockDim.x);
	curandState state;
	curand_init((unsigned long long)clock() + tId, 0, 0, &state);

	double rand1 = curand_uniform_double(&state) * (max - min) + min;
	*result = rand1;
	printf("randomCudaGenerator :: %f\n", rand1);
}

 float GenerateRandomFloatBetween(float a, float b)
{
	float res;
	std::uniform_real_distribution<double> distribution(a, b);
	res = distribution(generator);

	return res;
}
 int GenerateRandomIntegerBetween(int a, int b)
{
	std::uniform_int_distribution<int> gen(a, b);
	return gen(rng);
}

/*Operations Declarations*/
//First Operation
int hasBeenInfected(Agent* agent, vector<Agent*>* AgentList)
{
	if (agent->S != 0) {
		return agent->S;
	}
	// Part First Part
	float prob = GenerateRandomFloatBetween(0, 1);
	bool fp = prob <= agent->Pcon ? 1 : 0;
	bool a = 0;

	// Part: Calculate Alpha
	bool sp = 0;
	bool tp = 0;
	int sum = 0;

	for (Agent* agent_test : *AgentList)
	{
		if (agent->x == agent_test->x && agent->y == agent_test->y)
		{
			continue;
		}

		float distance = (float)sqrt(pow(agent_test->x - agent->x, 2) + pow(agent_test->y - agent->y, 2));
		sp = distance <= GlobalState::R;
		// Part: Calculate Beta
		tp = agent_test->S > 0 ? 1 : 0;
		sum += sp * tp;
	}
	a = sum >= 1 ? 1 : 0;


	return (fp * a) ? 1 : agent->S;
}

//Second Operation
int isShortMovement(Agent* agent)
{
	float prob = GenerateRandomFloatBetween(0, 1);
	return (prob <= agent->Psmo) ? 1 : 0;
}

int XMovement(Agent* agent)
{
	int s = isShortMovement(agent);
	int p = GlobalState::mapSize[1];
	float prob = GenerateRandomFloatBetween(0, 1);
	// printf("prob X 1: %f\n", prob);
	float longDistance = p * prob * (1 - s);
	prob = GenerateRandomFloatBetween(0, 1);
	// printf("prob X 2: %f\n", prob);
	float shortDistance = (agent->x + (2 * prob - 1) * GlobalState::lmax) * s;
	return shortDistance + longDistance;
}

int YMovement(Agent* agent)
{
	int s = isShortMovement(agent);
	int q = GlobalState::mapSize[0];
	float prob = GenerateRandomFloatBetween(0, 1);
	// printf("prob Y 1: %f\n", prob);
	float longDistance = q * prob * (1 - s);
	prob = GenerateRandomFloatBetween(0, 1);
	// printf("prob Y 2: %f\n", prob);
	float shortDistance = (agent->y + (2 * prob - 1) * GlobalState::lmax) * s;
	return shortDistance + longDistance;
}

bool willMove(Agent* agent)
{
	float prob = GenerateRandomFloatBetween(0, 1);
	return (prob <= agent->Pmov) ? 1 : 0;
}

//Third Operation
int hasBeenExternalInfected(Agent* agent)
{
	float prob = GenerateRandomFloatBetween(0, 1);
	//Part: Calculate First Part
	bool fp = prob <= agent->Pext ? 1 : 0;
	//Part: Calcula Epsilon
	bool sp = agent->S != 0 ? 0 : 1;
	//fp * sp == if fp or sp are 0 the condition is false
	bool finalB = (fp * sp) > 0;

	return (finalB) ? 1 : agent->S;
}

//Third Operation
__device__ int hasBeenExternalInfected_GPU(Agent* agent, float prob)
{
	//float prob = GenerateRandomFloatBetween(0, 1);
	//Part: Calculate First Part
	bool fp = prob <= agent->Pext ? 1 : 0;
	//Part: Calcula Epsilon
	bool sp = agent->S != 0 ? 0 : 1;
	//fp * sp == if fp or sp are 0 the condition is false
	bool finalB = (fp * sp) > 0;

	return (finalB) ? 1 : agent->S;
}

//Fourth Operation
__device__ int incubationTime(Agent* agent)
{
	int Tinc = agent->Tinc;
	if (agent->S > 0)
	{
		Tinc--;
	}
	return Tinc;
}

__device__ int hasSymptoms(Agent* agent)
{
	int S = -1;
	if (agent->Tinc > 0)
	{
		S = agent->S;
	}
	return S;
}

__device__ int recuperationTime(Agent* agent)
{
	int Trec = agent->Trec;
	if (agent->S == -1)
	{
		Trec--;
	}
	return Trec;
}

//Fifth Operation
 int isInRecuperation(Agent* agent)
{
	return (agent->S < 0) ? 1 : 0;
}
__device__ int isInRecuperation_GPU(Agent* agent)
{
	return (agent->S < 0) ? 1 : 0;
}

int isDead(Agent* agent)
{
	int o = isInRecuperation(agent);
	float prob = GenerateRandomFloatBetween(0, 1);
	int Pfat = (prob <= agent->Pfat) ? 1 : 0;
	return (Pfat * o > 0) ? -2 : agent->S;
}

__device__ int isDead_GPU(Agent* agent, float prob)
{
	int o = isInRecuperation_GPU(agent);
	//float prob = GenerateRandomFloatBetween(0, 1);
	int Pfat = (prob <= agent->Pfat) ? 1 : 0;
	return (Pfat * o > 0) ? -2 : agent->S;
}

int Clamp(int value, int low, int high)
{
	return value > high ? high : value < low ? low : value;
}

/*GlobalState Declarations*/
int GlobalState::mapSize[2] = { 500, 500 };
int GlobalState::Dmax = 60; //// DIAS
int GlobalState::Mmax = 10;
int GlobalState::lmax = 5;
int GlobalState::N = 10240; //// AGENTES 
// int GlobalState::N = 5000;
int GlobalState::infectedAgents = 0;
int GlobalState::R = 1;
/* Statistics*/
int GlobalState::curedAgents = 0;
int GlobalState::deadAgents = 0;
int GlobalState::patientZeroDay = 0;
int GlobalState::halfPopulationInfectedDay = 0;
int GlobalState::fullPopulationInfectedDay = 0;
int GlobalState::patientZeroCuredDay = 0;
int GlobalState::halfPopulationCuredDay = 0;
int GlobalState::fullPopulationCuredDay = 0;
int GlobalState::patientZeroDeadDay = 0;
int GlobalState::halfPopulationDeadDay = 0;
int GlobalState::fullPopulationDeadDay = 0;

float Agent::PconRange[2] = { 0.02, 0.03 };
float Agent::PextRange[2] = { 0.02, 0.03 };
float Agent::PfatRange[2] = { 0.007, 0.07 };
float Agent::PmovRange[2] = { 0.3, 0.5 };
float Agent::PsmoRange[2] = { 0.7, 0.9 };
int Agent::TincRange[2] = { 5, 6 };

/*Agent Constructor*/
Agent::Agent(int x, int y, int i)
{
	// Position
	Agent::id = i;
	Agent::x = x;
	Agent::y = y;
	// Assing the Infection Probability
	Agent::Pcon = GenerateRandomFloatBetween(Agent::PconRange[0], Agent::PconRange[1]);
	// Assign the External Infection Probability
	Agent::Pext = GenerateRandomFloatBetween(Agent::PextRange[0], Agent::PextRange[1]);
	//Assign the Mortality Probability
	Agent::Pfat = GenerateRandomFloatBetween(Agent::PfatRange[0], Agent::PfatRange[1]);
	// Assign the Movement Probability
	Agent::Pmov = GenerateRandomFloatBetween(Agent::PmovRange[0], Agent::PmovRange[1]);
	// Assign the Small Movement Probability
	Agent::Psmo = GenerateRandomFloatBetween(Agent::PsmoRange[0], Agent::PsmoRange[1]);
	// Assign the Incubation Time
	Agent::Tinc = GenerateRandomFloatBetween(Agent::TincRange[0], Agent::TincRange[1]);
};

__global__ void Rule345(Agent* AgentList) {

	int min_1 = 0;
	int max_1 = 1;
	int min_2 = 0;
	int max_2 = 1;
	curandState state;
	int tId = threadIdx.x + (blockIdx.x * blockDim.x);
	curand_init((unsigned long long)clock() + tId, 0, 0, &state);
	double rand1 = curand_uniform_double(&state) * (max_1 - min_1) + min_1;
	double rand2 = curand_uniform_double(&state) * (max_2 - min_2) + min_2;
	int gId = blockIdx.x * blockDim.x + threadIdx.x;

	Agent agent = AgentList[gId];

	//Aplicar Regla 3
	//agent->S = hasBeenExternalInfected(agent);
	agent.S = hasBeenExternalInfected_GPU(&agent, (float)rand1);

	//Aplicar Regla 4
	agent.Tinc = incubationTime(&agent);
	agent.S = hasSymptoms(&agent);
	agent.Trec = recuperationTime(&agent);
	if (agent.Trec <= 0) {
		agent.Pcon = -1;
		agent.Pext = -1;
		agent.Pfat = -1;
		agent.S = 0;

		if (!agent.hasBeenAccountedFor) {
			agent.hasBeenAccountedFor = true;
		}

	}
	//Aplicar Regla 5
	agent.S = isDead_GPU(&agent, (float)rand2);

	AgentList[gId] = agent;
}


/**/
int main()
{
	clock_t total_start_CPU = clock();

	/* 1. Map Generation and Setup */
	int size0 = 500;
	int size1 = 500;
	vector<int> historyInfectedPerDay;
	vector<int> historyCuredPerDay;
	vector<int> historyDeadPerDay;

	//vector<Agent*> AgentList;
	//Agent** AgentList = new Agent*[GlobalState::N];
	Agent* AgentList = new Agent[GlobalState::N];
	Agent **AgentMap = new Agent*[500 * 500];


		for (int i = 0; i < size0; i++)
		{
			for (int j = 0; j < size1; j++)
			{
				AgentMap[i * 500 + j] = NULL;
			}
		}


	/* 2. Generate Agents on Map */
	int leftAgents = GlobalState::N;
	int i = 0;
	while (leftAgents > 0)
	{
		int x = GenerateRandomIntegerBetween(0, 499);
		int y = GenerateRandomIntegerBetween(0, 499);

		if (AgentMap[y * 500 + x] != NULL)
		{
			continue;
		}

		Agent* agent = new Agent(x, y, i);
		// printf("(%d,%d)\n", agent->x, agent->y);
		AgentMap[y * 500 + x] = agent;

		AgentList[i] = *agent;
		leftAgents--;
		i++;
	}

	/* Operation Phase*/
	int currentDay = 0;
	//Mientras que el d�a actual de simulaci�n sea menor que dMax
	while (currentDay < GlobalState::Dmax)
	{
		int infectedToday = 0;
		int killedToday = 0;
		int curedToday = 0;

		int currentMovement = 0;
		//Mientras que el movimiento actual sea menor que mMax
		while (currentMovement < GlobalState::Mmax)
		{
			//Para todos los agentes
			for (int i = 0; i < GlobalState::N; i++)
			{
				Agent* agent = &AgentList[i];
				int previousState = agent->S;

				//Aplicar Regla 1
				vector<Agent*> Neighbours;
				//if (agent->y > 0 && AgentMap[agent->y - 1][agent->x] != NULL)
				if (agent->y > 0 && AgentMap[(agent->y - 1)*500+(agent->x)] != NULL)
				{
					Neighbours.push_back(AgentMap[(agent->y - 1) * 500 + (agent->x)]);
				}
				if (agent->x > 0 && AgentMap[(agent->y)*500 + (agent->x - 1)] != NULL)
				{
					Neighbours.push_back(AgentMap[(agent->y) * 500 + (agent->x - 1)]);
				}
				if (agent->x < size0 - 1 && AgentMap[(agent->y) * 500 + (agent->x + 1)] != NULL)
				{
					Neighbours.push_back(AgentMap[(agent->y) * 500 + (agent->x + 1)]);
				}
				if (agent->y < size1 - 1 && AgentMap[(agent->y + 1)*500 + (agent->x)] != NULL)
				{
					Neighbours.push_back(AgentMap[(agent->y + 1) * 500 + (agent->x)]);
				}
				agent->S = hasBeenInfected(agent, &Neighbours);


				//Aplicar Regla 2
				if (willMove(agent))
				{
					int x;
					int y;

					do
					{
						x = XMovement(agent);
						y = YMovement(agent);

						x = Clamp(x, 0, size0 - 1);
						y = Clamp(y, 0, size1 - 1);
					} while (AgentMap[y * 500 + x] != NULL);

					//Se elimina de la vieja posici�n
					AgentMap[agent->y * 500 + agent->x] = NULL;

					//Se a�ade a la nueva posici�n
					agent->x = x;
					agent->y = y;
					AgentMap[y * 500 + x] = agent;
				}

				if (agent->S != previousState) {
					switch (agent->S) {
					case 1: //Infected
					  // printf("Agent(%d, %d) has been infected on Internal!\n", agent->x, agent->y);
						if (GlobalState::infectedAgents == 0)
						{
							printf("\t\tPatient Zero has been found\n");
							GlobalState::patientZeroDay = currentDay;
						}
						GlobalState::infectedAgents++;
						infectedToday++;
						break;
					}
				}
			}

			currentMovement++;
		}

		/* KERNEL 345 */

		Agent* DEV_AgentList;
		Agent* DEV_Result_AgentList;
		DEV_Result_AgentList = (Agent*)malloc(GlobalState::N * sizeof(Agent));

		cudaMalloc((void**)&DEV_AgentList, GlobalState::N * sizeof(Agent));
		check_CUDA_error("Malloc DEV_AgentList :: 345");

		cudaMemcpy(DEV_AgentList, AgentList, GlobalState::N * sizeof(Agent), cudaMemcpyHostToDevice);
		check_CUDA_error("Memcpy Data HOST :: DEV 345");

		dim3 grid(10);
		dim3 block(1024);

		Rule345 << <grid, block >> > (DEV_AgentList);
		check_CUDA_error("Rule345 ::");

		cudaMemcpy(DEV_Result_AgentList, DEV_AgentList, GlobalState::N * sizeof(Agent), cudaMemcpyDeviceToHost);
		check_CUDA_error("Memcpy Data DEV :: HOST 345");

		int infectedAgentsGPU = 0;
		int curedAgentsGPU = 0;
		int deadAgentsGPU = 0;

		int totalInfected = 0;

		for (int i = 0; i < GlobalState::N; i++) {
			Agent* gpu_agent = &DEV_Result_AgentList[i];

			//printf("[Pos:%d] Agent [ID:%d] (%d, %d) -> Status: %d\n", i, gpu_agent->id, gpu_agent->x, gpu_agent->y, gpu_agent->S);
			switch (gpu_agent->S)
			{
			case 1: 
				//Infected

				infectedAgentsGPU++;
				totalInfected++;
				break;
			case -1:
				totalInfected++;
				break;
			case -2:
				//Dead
				gpu_agent->Pcon = -1;
				gpu_agent->Pext = -1;
				gpu_agent->Pfat = -1;
				gpu_agent->S = 0;
				if (!gpu_agent->hasBeenAccountedFor) {
					deadAgentsGPU++;
					gpu_agent->hasBeenAccountedFor = true;
					gpu_agent->hasDied = true;
				}
				totalInfected++;
				break;
			case 0:
				if (gpu_agent->hasBeenAccountedFor && !gpu_agent->hasDied) {
					curedAgentsGPU++;
					totalInfected++;
				}
				break;
			}
		}
		
		/*printf("\tB:Infected: %d\n", infectedToday);
		printf("\tB:Dead: %d\n", killedToday);
		printf("\tB:Cured: %d\n\n", curedToday);

		printf("\tTotalInfectedGPU: %d\n", totalInfected);
		printf("\tTotalDeadGPU: %d\n", deadAgentsGPU);
		printf("\tTotalCuredGPU: %d\n\n", curedAgentsGPU);

		printf("\tGlobalInfected: %d\n", GlobalState::infectedAgents);
		printf("\tGlobalDead: %d\n", GlobalState::deadAgents);
		printf("\tGlobalCured: %d\n\n", GlobalState::curedAgents);

		printf("\tCalcInfectedToday: %d\n", abs(GlobalState::infectedAgents - totalInfected));
		printf("\tCalcDeadToday: %d\n", abs(GlobalState::deadAgents - deadAgentsGPU));
		printf("\tCalcCuredToday: %d\n\n", abs(GlobalState::curedAgents - curedAgentsGPU));*/

		infectedToday += totalInfected - GlobalState::infectedAgents;
		if (infectedToday < 0) {
			infectedToday = 0;
		}
		killedToday += deadAgentsGPU - GlobalState::deadAgents;
		if (killedToday < 0) {
			killedToday = 0;
		}
		curedToday += curedAgentsGPU - GlobalState::curedAgents;
		if (curedToday < 0) {
			curedToday = 0;
		}

		historyInfectedPerDay.push_back(infectedToday);
		historyDeadPerDay.push_back(killedToday);
		historyCuredPerDay.push_back(curedToday);

		// printf("\tInfected: %d\n", infectedToday);
		// printf("\tDead: %d\n", killedToday);
		// printf("\tCured: %d\n", curedToday);


		int sumInfected = 0;
		int sumKilled = 0;
		int sumCured = 0;
		for (int i = 0; i < historyInfectedPerDay.size(); i++) {
			sumInfected += historyInfectedPerDay[i];
		}
		for (int i = 0; i < historyDeadPerDay.size(); i++) {
			sumKilled += historyDeadPerDay[i];
		}
		for (int i = 0; i < historyCuredPerDay.size(); i++) {
			sumCured += historyCuredPerDay[i];
		}

		GlobalState::infectedAgents = sumInfected;
		GlobalState::deadAgents = sumKilled;
		GlobalState::curedAgents = sumCured;

		AgentList = DEV_Result_AgentList;
		currentDay++;
	}	
	clock_t total_end_CPU = clock();
	float total_elapsedTime_CPU = total_end_CPU - total_start_CPU;

	int maxInfected = GlobalState::infectedAgents;
	int sumInfected = 0;
	int sumDead = 0;
	int sumCured = 0;

	bool hasFirstInfectedBeenDetected = false;
	bool hasFirstDeadBeenDetected = false;
	bool hasFirstCuredBeenDetected = false;
	
	bool hasHalfPopulationInfectedBeenDetected = false;
	bool hasHalfPopulationDeadBeenDetected = false;
	bool hasHalfPopulationCuredBeenDetected = false;

	bool hasAllPopulationInfectedBeenDetected = false;
	bool hasAllPopulationDeadBeenDetected = false;
	bool hasAllPopulationCuredBeenDetected = false;

	for(int i = 0; i < GlobalState::Dmax; i++){
		int curDayInfected  = historyInfectedPerDay[i];
		int curDayDead  = historyDeadPerDay[i];
		int curDayCured  = historyCuredPerDay[i];

		sumInfected += curDayInfected;
		sumDead += curDayDead;
		sumCured += curDayCured;

		/* Patients Zero */    
		if(!hasFirstInfectedBeenDetected){
		if(curDayInfected > 0){
			GlobalState::patientZeroDay = i+1;
			hasFirstInfectedBeenDetected = true;
		}
		}
		if(!hasFirstDeadBeenDetected){
		if(curDayDead > 0){
			GlobalState::patientZeroDeadDay = i+1;
			hasFirstDeadBeenDetected = true;
		}
		}
		if(!hasFirstCuredBeenDetected){
		if(curDayCured > 0){
			GlobalState::patientZeroCuredDay = i+1;
			hasFirstCuredBeenDetected = true;
		}
		}

		/* Half population */
		if(!hasHalfPopulationInfectedBeenDetected){
		if(sumInfected > (maxInfected/2) ){
			GlobalState::halfPopulationInfectedDay = i+1;
			hasHalfPopulationInfectedBeenDetected = true;
		}
		}
		if(!hasHalfPopulationDeadBeenDetected){
		if(sumDead > (maxInfected/2) ){
			GlobalState::halfPopulationDeadDay = i+1;
			hasHalfPopulationDeadBeenDetected = true;
		}
		}
		if(!hasHalfPopulationCuredBeenDetected){
		if(sumCured > (maxInfected/2) ){
			GlobalState::halfPopulationCuredDay = i+1;
			hasHalfPopulationCuredBeenDetected = true;
		}
		}

		/* All population */
		if(!hasAllPopulationInfectedBeenDetected){
		if(sumInfected >= maxInfected ){
			GlobalState::fullPopulationInfectedDay = i+1;
			hasAllPopulationInfectedBeenDetected = true;
		}
		}
		if(!hasAllPopulationDeadBeenDetected){
		if(sumDead >= maxInfected ){
			GlobalState::fullPopulationDeadDay = i+1;
			hasAllPopulationDeadBeenDetected = true;
		}
		}
		if(!hasAllPopulationCuredBeenDetected){
		if(sumCured >= maxInfected ){
			GlobalState::fullPopulationCuredDay = i+1;
			hasAllPopulationCuredBeenDetected = true;
		}
		}

	}

	/* END */
	printf("Phase End\n");
	printf("\tTotal Time GPU: %f ms.\n", total_elapsedTime_CPU);
	printf("===\nTotal Agents: %d\n===\n", GlobalState::N);
	printf("Total Infected Agents: %d\n", GlobalState::infectedAgents);
	printf("Total Dead Agents: %d\n", GlobalState::deadAgents);
	printf("Total Cured Agents: %d\n===\n", GlobalState::curedAgents);
	printf("Day of First Agent Infection: %d\n", GlobalState::patientZeroDay);
	printf("Day of First Agent Dead: %d\n", GlobalState::patientZeroDeadDay);
	printf("Day of First Agent Cured: %d\n===\n", GlobalState::patientZeroCuredDay);
	printf("Day of Half Population Infection: %d\n", GlobalState::halfPopulationInfectedDay);
	printf("Day of Half Population Dead: %d\n", GlobalState::halfPopulationDeadDay);
	printf("Day of Half Population Cured: %d\n===\n", GlobalState::halfPopulationCuredDay);
	printf("Day of Full Population Infection: %d\n", GlobalState::fullPopulationInfectedDay);
	printf("Day of Full Population Dead: %d\n", GlobalState::fullPopulationDeadDay);
	printf("Day of Full Population Cured: %d\n===\n", GlobalState::fullPopulationCuredDay);

	printf("Agents infected per Day (History):");
	for (int val : historyInfectedPerDay)
	{
		printf(" %d", val);
	}
	printf("\n");
	printf("Agents killed per Day (History):");
	for (int val : historyDeadPerDay)
	{
		printf(" %d", val);
	}
	printf("\n");
	printf("Agents cured per Day (History):");
	for (int val : historyCuredPerDay)
	{
		printf(" %d", val);
	}
	printf("\n");
	return 0;
}


/*Authors:
- Alan Enrique Maldonado Navarro
- Guillermo Gonzalez Mena

Repository: https://github.com/DrN3MESiS/covid-19-spread-simulator
*/
