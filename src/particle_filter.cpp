/*
 * particle_filter.cpp
 *
 *  Created on: Dec 12, 2016
 *      Author: Tiffany Huang
 */

#include <random>
#include <algorithm>
#include <iostream>
#include <numeric>
#include <math.h> 
#include <iostream>
#include <sstream>
#include <string>
#include <iterator>

#include "particle_filter.h"

#define EPS 0.00001

using namespace std;

void ParticleFilter::init(double x, double y, double theta, double std[]) {
	// TODO: Set the number of particles. Initialize all particles to first position (based on estimates of 
	//   x, y, theta and their uncertainties from GPS) and all weights to 1. 
	// Add random Gaussian noise to each particle.
	// NOTE: Consult particle_filter.h for more information about this method (and others in this file).

        // This line creates a normal (Gaussian) distribution for x.
	normal_distribution<double> dist_x(x, std[0]);
        
        // Create normal distributions for y and theta.
	normal_distribution<double> dist_y(y, std[1]);
        
        normal_distribution<double> dist_theta(theta, std[2]);

        num_particles = 100;
        default_random_engine gen;
        for (int i = 0 ; i<num_particles;i++)
        {
            Particle P;
            P.id = i;
            P.x = dist_x(gen);
            P.y = dist_y(gen);
            P.theta = dist_theta(gen);
            P.weight = 1;
            particles.push_back(P);
        }
        is_initialized = true;
}

void ParticleFilter::prediction(double delta_t, double std_pos[], double velocity, double yaw_rate) {
	// TODO: Add measurements to each particle and add random Gaussian noise.
	// NOTE: When adding noise you may find std::normal_distribution and std::default_random_engine useful.
	//  http://en.cppreference.com/w/cpp/numeric/random/normal_distribution
	//  http://www.cplusplus.com/reference/random/default_random_engine/
        default_random_engine gen;
        for (int i =0; i<num_particles;i++)
        {
            if(yaw_rate!=0)//motion model yaw rate != 0
            {
                particles[i].x = particles[i].x + velocity*(sin(particles[i].theta+yaw_rate*delta_t)-sin(particles[i].theta))/yaw_rate;
                particles[i].y = particles[i].y + velocity*(cos(particles[i].theta)-cos(particles[i].theta+yaw_rate*delta_t))/yaw_rate;
                particles[i].theta = particles[i].theta + yaw_rate*delta_t;
            }
            else //motion model yaw rate == 0
            {
                particles[i].x = particles[i].x + velocity*delta_t*cos(particles[i].theta);
                particles[i].y = particles[i].y + velocity*delta_t*sin(particles[i].theta);
                particles[i].theta = particles[i].theta;
            }
            //add noise
            normal_distribution<double> dist_x(particles[i].x, std_pos[0]);
            normal_distribution<double> dist_y(particles[i].y, std_pos[1]);
            normal_distribution<double> dist_theta(particles[i].theta, std_pos[2]);
            
            particles[i].x = dist_x(gen);
            particles[i].y = dist_y(gen);
            particles[i].theta = dist_theta(gen);
        }
      
}

void ParticleFilter::dataAssociation(std::vector<LandmarkObs> predicted, std::vector<LandmarkObs>& observations) {
	// TODO: Find the predicted measurement that is closest to each observed measurement and assign the 
	//   observed measurement to this particular landmark.
	// NOTE: this method will NOT be called by the grading code. But you will probably find it useful to 
	//   implement this method and use it as a helper during the updateWeights phase.

        for(unsigned int i=0;i<observations.size();i++)
        {
            double dist_min = numeric_limits<double>::max();
            int mapId = -1;
            for(unsigned int j=0;j<predicted.size();j++)
            {    // find the min one
                 if(dist_min >= dist(observations[i].x,observations[i].y,predicted[j].x,predicted[j].y))
                 {
                      dist_min = dist(observations[i].x,observations[i].y,predicted[j].x,predicted[j].y);
                      mapId = predicted[j].id;
                 }
            }
            //get the associated id
            observations[i].id = mapId;
        }

}

void ParticleFilter::updateWeights(double sensor_range, double std_landmark[], 
		const std::vector<LandmarkObs> &observations, const Map &map_landmarks) {
	// TODO: Update the weights of each particle using a mult-variate Gaussian distribution. You can read
	//   more about this distribution here: https://en.wikipedia.org/wiki/Multivariate_normal_distribution
	// NOTE: The observations are given in the VEHICLE'S coordinate system. Your particles are located
	//   according to the MAP'S coordinate system. You will need to transform between the two systems.
	//   Keep in mind that this transformation requires both rotation AND translation (but no scaling).
	//   The following is a good resource for the theory:
	//   https://www.willamette.edu/~gorr/classes/GeneralGraphics/Transforms/transforms2d.htm
	//   and the following is a good resource for the actual equation to implement (look at equation 
	//   3.33
	//   http://planning.cs.uiuc.edu/node99.html
        for(int i=0; i<num_particles;i++)
        {
            
            vector<double> sense_x;
            vector<double> sense_y;
            std::vector<LandmarkObs> obsers_m;// observations in map cordinate
            for(unsigned int j=0; j<observations.size();j++)
            {
                //transfer the observation from particle view to map cordinate
                LandmarkObs o_m;
                o_m.x = particles[i].x + observations[j].x*cos(particles[i].theta) - sin(particles[i].theta)*observations[j].y;
                o_m.y = particles[i].y + observations[j].x*sin(particles[i].theta) + cos(particles[i].theta)*observations[j].y;
                obsers_m.push_back(o_m);
                sense_x.push_back(o_m.x);
                sense_y.push_back(o_m.y);
            }
            
            std::vector<LandmarkObs> predicted_l;
            for(unsigned int k =0; k<map_landmarks.landmark_list.size();k++)
            {
                //find the in range landmark
                if(dist(particles[i].x,particles[i].y,map_landmarks.landmark_list[k].x_f,map_landmarks.landmark_list[k].y_f)<=sensor_range)
                {
                    LandmarkObs predict_m;
                    predict_m.id = map_landmarks.landmark_list[k].id_i;
                    predict_m.x = map_landmarks.landmark_list[k].x_f;
                    predict_m.y = map_landmarks.landmark_list[k].y_f;
                    predicted_l.push_back(predict_m);
                }
            }
            dataAssociation(predicted_l,obsers_m);
            particles[i].weight = 1.0;
            vector<int> associations;
            for(unsigned int j=0; j<obsers_m.size();j++)
            {
                //calculate the multivariate-gaussian probability
                double P_xy,gauss_norm,exponent1,exponent2;
                gauss_norm = 1.0/(2*M_PI*std_landmark[0]*std_landmark[1]);
                //the landmark index should be from 0, need minus 1
                exponent1 = pow((obsers_m[j].x - map_landmarks.landmark_list[obsers_m[j].id-1].x_f),2)*1.0/(2*std_landmark[0]*std_landmark[0]);
                exponent2 = pow((obsers_m[j].y - map_landmarks.landmark_list[obsers_m[j].id-1].y_f),2)*1.0/(2*std_landmark[1]*std_landmark[1]);
                P_xy = gauss_norm*exp(-(exponent1+exponent2));
                if(P_xy ==0)
                {
                    particles[i].weight = particles[i].weight*EPS;//prevent from zero weight
                }
                else
                {
                    particles[i].weight = particles[i].weight*P_xy;
                }
                associations.push_back(obsers_m[j].id);
            }
            
            SetAssociations(particles[i],associations,sense_x,sense_y);//set associations
        }

}

void ParticleFilter::resample() {
	// TODO: Resample particles with replacement with probability proportional to their weight. 
	// NOTE: You may find std::discrete_distribution helpful here.
	//   http://en.cppreference.com/w/cpp/numeric/random/discrete_distribution
        default_random_engine gen;
        std::vector<double> p_w;
        std::vector<Particle> re_particles;
        for(int i=0; i<num_particles;i++)
        {
             p_w.push_back(particles[i].weight);
        }
        discrete_distribution<int> distribution(p_w.begin(),p_w.end());//use discrete_distribution to resample
        for(int i=0; i<num_particles;i++)
        {
             re_particles.push_back(particles[distribution(gen)]);
        }
        particles = re_particles;
}

Particle ParticleFilter::SetAssociations(Particle& particle, const std::vector<int>& associations, 
                                     const std::vector<double>& sense_x, const std::vector<double>& sense_y)
{
    //particle: the particle to assign each listed association, and association's (x,y) world coordinates mapping to
    // associations: The landmark id that goes along with each listed association
    // sense_x: the associations x mapping already converted to world coordinates
    // sense_y: the associations y mapping already converted to world coordinates

    particle.associations= associations;
    particle.sense_x = sense_x;
    particle.sense_y = sense_y;

    return particle;
}

string ParticleFilter::getAssociations(Particle best)
{
	vector<int> v = best.associations;
	stringstream ss;
    copy( v.begin(), v.end(), ostream_iterator<int>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length()-1);  // get rid of the trailing space
    return s;
}
string ParticleFilter::getSenseX(Particle best)
{
	vector<double> v = best.sense_x;
	stringstream ss;
    copy( v.begin(), v.end(), ostream_iterator<float>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length()-1);  // get rid of the trailing space
    return s;
}
string ParticleFilter::getSenseY(Particle best)
{
	vector<double> v = best.sense_y;
	stringstream ss;
    copy( v.begin(), v.end(), ostream_iterator<float>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length()-1);  // get rid of the trailing space
    return s;
}
