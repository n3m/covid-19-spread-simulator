#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <cstring>
#include <math.h>
#include <time.h>
#include <random>
#include <chrono>
#include <vector>
#include <string>
#include <sstream>

#include "classes.cpp"

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
	int res = 0;
	// Part First Part
	float prob = GenerateRandomFloatBetween(0, 1);
	bool fp = prob <= agent->Pcon ? 1 : 0;
	bool a = 0;

	// Part: Calculate Alpha
	bool sp = 0;
	bool tp = 0;
	int validateFn = 0;
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

//Fourth Operation
int incubationTime(Agent* agent)
{
	int Tinc = agent->Tinc;
	if (agent->S > 0)
	{
		Tinc--;
	}
	return Tinc;
}

int hasSymptoms(Agent* agent)
{
	int S = -1;
	if (agent->Tinc > 0)
	{
		S = agent->S;
	}
	return S;
}

int recuperationTime(Agent* agent)
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

int isDead(Agent* agent)
{
	int o = isInRecuperation(agent);
	float prob = GenerateRandomFloatBetween(0, 1);
	int Pfat = (prob <= agent->Pfat) ? 1 : 0;
	return (Pfat * o > 0) ? -2 : agent->S;
}

int Clamp(int value, int low, int high)
{
	return value > high ? high : value < low ? low : value;
}

/*GlobalState Declarations*/
int GlobalState::mapSize[2] = { 500, 500 };
int GlobalState::Dmax = 60;
int GlobalState::Mmax = 10;
int GlobalState::lmax = 5;
int GlobalState::N = 10240;
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
Agent::Agent(int x, int y)
{
	// Position
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

	vector<Agent*> AgentList;
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
	while (leftAgents > 0)
	{
		int x = GenerateRandomIntegerBetween(0, 499);
		int y = GenerateRandomIntegerBetween(0, 499);

		if (AgentMap[y*500+x] != NULL)
		{
			continue;
		}

		Agent* agent = new Agent(x, y);
		// printf("(%d,%d)\n", agent->x, agent->y);
		AgentMap[y * 500 + x] = agent;
		AgentList.push_back(agent);
		leftAgents--;
	}

	/* Operation Phase*/
	int currentDay = 0;
	//Mientras que el día actual de simulación sea menor que dMax
	while (currentDay < GlobalState::Dmax)
	{
		int infectedToday = 0;
		int killedToday = 0;
		int curedToday = 0;
		printf("== Day %d (%d)\n ", (currentDay + 1), GlobalState::infectedAgents);
		int currentMovement = 0;
		//Mientras que el movimiento actual sea menor que mMax
		while (currentMovement < GlobalState::Mmax)
		{
			//Para todos los agentes
			for (int i = 0; i < AgentList.size(); i++)
			{
				Agent* agent = AgentList[i];
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

					//Se elimina de la vieja posición
					AgentMap[agent->y * 500 + agent->x] = NULL;

					//Se añade a la nueva posición
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


		for (int i = 0; i < AgentList.size(); i++)
		{
			Agent* agent = AgentList[i];
			int previousState = agent->S;

			//Aplicar Regla 3
			agent->S = hasBeenExternalInfected(agent);

			//Aplicar Regla 4
			agent->Tinc = incubationTime(agent);
			agent->S = hasSymptoms(agent);
			agent->Trec = recuperationTime(agent);
			if (agent->Trec <= 0) {
				agent->Pcon = -1;
				agent->Pext = -1;
				agent->Pfat = -1;
				agent->S = 0;

				if (!agent->hasBeenAccountedFor) {
					if (GlobalState::curedAgents == 0)
					{
						printf("\t\tPatient Zero has been found\n");
						GlobalState::patientZeroCuredDay = currentDay;
					}
					GlobalState::curedAgents++;
					curedToday++;
					agent->hasBeenAccountedFor = true;
				}
			}
			//Aplicar Regla 5
			agent->S = isDead(agent);

			if (agent->S != previousState) {
				switch (agent->S) {
				case 1: //Infected
				// printf("Agent(%d, %d) has been infected on External!\n", agent->x, agent->y);
					if (GlobalState::infectedAgents == 0)
					{
						printf("\t\tPatient Zero has been found\n");
						GlobalState::patientZeroDay = currentDay;
					}
					GlobalState::infectedAgents++;
					infectedToday++;
					break;
				case -2: // Dead
					agent->Pcon = -1;
					agent->Pext = -1;
					agent->Pfat = -1;
					agent->S = 0;
					agent->hasBeenAccountedFor = true; //TEST

					if (GlobalState::deadAgents == 0)
					{
						printf("\t\tPatient Zero has died\n");
						GlobalState::patientZeroDeadDay = currentDay;
					}
					GlobalState::deadAgents++;
					killedToday++;
					break;
				}
			}
		}


		historyCuredPerDay.push_back(curedToday);
		historyDeadPerDay.push_back(killedToday);
		historyInfectedPerDay.push_back(infectedToday);

		printf("\tInfected: %d\n", infectedToday);
		printf("\tDead: %d\n", killedToday);
		printf("\tCured: %d\n", curedToday);

		curedToday = 0;
		killedToday = 0;
		infectedToday = 0;

		currentDay++;
	}

  int maxInfected = GlobalState::infectedAgents;
  int maxDead = GlobalState::deadAgents;
  int maxCured = GlobalState::curedAgents;
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
      if(sumDead > (maxDead/2) ){
        GlobalState::halfPopulationDeadDay = i+1;
        hasHalfPopulationDeadBeenDetected = true;
      }
    }
    if(!hasHalfPopulationCuredBeenDetected){
      if(sumCured > (maxCured/2) ){
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
      if(sumDead >= maxDead ){
        GlobalState::fullPopulationDeadDay = i+1;
        hasAllPopulationDeadBeenDetected = true;
      }
    }
    if(!hasAllPopulationCuredBeenDetected){
      if(sumCured >= maxCured ){
        GlobalState::fullPopulationCuredDay = i+1;
        hasAllPopulationCuredBeenDetected = true;
      }
    }

  }

	/* END */
	printf("Phase End\n");
	clock_t total_end_CPU = clock();
	float total_elapsedTime_CPU = total_end_CPU - total_start_CPU;
	printf("\tTotal Time CPU: %f ms.\n", total_elapsedTime_CPU);
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
