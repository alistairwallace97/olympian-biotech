
/*Code for collecting sensor data, currently used for collecting HR data, bandpass filter already included
 * use a serial logger to store the data into text file, then change to csv file so that can be imported into python or matlab projects
*/
#include <Arduino.h>
#include <SPI.h>
#include "Adafruit_BLE.h"
#include "Adafruit_BluefruitLE_SPI.h"
#include "Adafruit_BluefruitLE_UART.h"
#include<Wire.h>
#include <Filters.h>

//  Variables
int PulseSensorPurplePin = 0;        // Pulse Sensor PURPLE WIRE connected to ANALOG PIN 0
int MovementNoisePin = A1;
int LED13 = 13;   //  The on-board Arduion LED
const int PIEZO_PIN1 = A4; // Piezo output
const int PIEZO_PIN2 = A5; // Piezo output
const int EMG_PIN1 = A2;
const int EMG_PIN2 = A3;
const int SW_PIN = 5;

int buttonstate = 0;

struct recentMax {

  float value;

  int counter;

};




const int MPU6050_addr=0x68; // holds the incoming raw data. Signal value can range from 0-1024
int16_t AccX,AccY,AccZ,Temp,GyroX,GyroY,GyroZ,AccX_init,AccY_init,AccZ_init,Temp_init,GyroX_init,GyroY_init,GyroZ_init;
bool initalize=1;
int16_t Signal;
int Threshold = 550;            // Determine which Signal to "count as a beat", and which to ingore.
int count = 0;
unsigned long prev = 0;

/*//global variable for  low pass filter
    FilterTwoPole filterTwoLowpass;               // create a two pole Lowpass filter
    filterTwoLowpass.setAsFilter( LOWPASS_BUTTERWORTH, hrHighCutoff );
    FilterOnePole filterOneHighpass( HIGHPASS, hrLowCutoff );  // create a one pole (RC) highpass filter
    int HrSignal;   */                              // holds the incoming raw data. Signal value can range from 0-1024


//setup filters
 float hrLowCutoff = 0.5;                      // 30 bpm

    float hrHighCutoff = 4.166666;                // 250 bpm

    float bandpassCurrent = 0;    

    float bandpassNoiseCurrent = 0; 

    float TMinusOneHr = 0;

    float TMinusTwoHr = 0;



    // standard Lowpass, set to the corner frequency

    FilterTwoPole filterTwoLowpass;               // create a two pole Lowpass filter

    //filterTwoLowpass.setAsFilter( LOWPASS_BUTTERWORTH, hrHighCutoff );

    FilterOnePole filterOneHighpass( HIGHPASS, hrLowCutoff );  // create a one pole (RC) highpass filter

    

    FilterTwoPole filterTwoLpNoise;               // create a two pole Lowpass filter

    //filterTwoLpNoise.setAsFilter( LOWPASS_BUTTERWORTH, hrHighCutoff );

    FilterOnePole filterOneHpNoise( HIGHPASS, hrLowCutoff );  // create a one pole (RC) highpass filter

    

    int HrSignal;                                 // holds the incoming raw data. Signal value can range from 0-1024

    int HrNoiseSignal;                                 // holds the incoming raw data. Signal value can range from 0-1024



    int counter = 0;
    bool firsttime = true;

   


// The SetUp Function:
void setup() {
    Wire.begin();
    Wire.beginTransmission(MPU6050_addr);
    Wire.write(0x6B);
    Wire.write(0);
    Wire.endTransmission(true);
    pinMode(LED13,OUTPUT);         // pin that will blink to your heartbeat!
    pinMode(SW_PIN,INPUT);
    Serial.begin(9600);         // Set's up Serial Communication at certain speed.
}

// The Main Loop Function
void loop() {
    Serial.println("hi");
     filterTwoLowpass.setAsFilter( LOWPASS_BUTTERWORTH, hrHighCutoff );
     filterTwoLpNoise.setAsFilter( LOWPASS_BUTTERWORTH, hrHighCutoff );
    recentMax maxOne;

    maxOne.value = 0;

    maxOne.counter = 0;

    recentMax maxTwo;

    maxTwo.value = 0;

    maxTwo.counter = 0;

    recentMax maxThree;

    maxThree.value = 0;

    maxThree.counter = 0;

    int maxValueTime = 180;

    float thresholdRatio = 0.70;

    float threshold = 0.00;

    bool aboveThreshold = false;

    unsigned long lastTime;

    unsigned long latestTime;

    unsigned long firstTime;

    float beatCounter = 0.00;

    float instHr = 0.00;

    float avgHr = 0.00;
    firsttime = false;
    Serial.println("cough state, EMG1, EMG2, Vib1, Vib2, Ax, Ay, Az, Gx, Gy, Gz, HR bandbass, InstantHR, Avg HR, People");

 while(true){
  //---------------acccelerometer setup----------------
  unsigned long StartTime = micros();
  Wire.beginTransmission(MPU6050_addr);
  Wire.write(0x3B);
  Wire.endTransmission(false);
  Wire.requestFrom(MPU6050_addr,14,true);

  //-----------Buttons------------------
  buttonstate = digitalRead(SW_PIN);
  if (buttonstate == LOW){
    //Serial.print("cough state: ");
   // Serial.print("1");
    //Serial.print(",");
  }
  else {
    //Serial.print("cough state: ");
   // Serial.print("0");
    //Serial.print(",");
  }

  //------------EMG sensor------------------
  int EMGADC1 = analogRead(EMG_PIN1);
  int EMGADC2 = analogRead(EMG_PIN2);
  //Serial.print("EMGdata: ");
  Serial.println(EMGADC1);
  /*Serial.print(",");
  Serial.print(EMGADC2);
  Serial.print(",");*/
  //----------------------------------------

  //----------------vibration sensor----------------------------
  // Read Piezo ADC value in, and convert it to a voltage
 /* int piezoADC1 = analogRead(PIEZO_PIN1);
  float piezoV1 = piezoADC1 / 1023.0 * 100;
  Serial.print(piezoV1); // Print the voltage.
  Serial.print(",");
  int piezoADC2 = analogRead(PIEZO_PIN2);
  float piezoV2 = piezoADC2 / 1023.0 * 100;
  Serial.print(piezoV2); // Print the voltage.
  Serial.print(",");*/
  //----------------------------------------------------------------

  //---------accelerometer sensor----------------------
  AccX=Wire.read()<<8|Wire.read();
  AccY=Wire.read()<<8|Wire.read();
  AccZ=Wire.read()<<8|Wire.read();
  //Temp=Wire.read()<<8|Wire.read();
  GyroX=Wire.read()<<8|Wire.read();
  GyroY=Wire.read()<<8|Wire.read();
  GyroZ=Wire.read()<<8|Wire.read();
//------------------------------------------------------
  
  //----------apply bandpass filter for HR signals--------------
  Signal = analogRead(PulseSensorPurplePin);  // Read the PulseSensor's value.
                                                  // Assign this value to the "Signal" variable.
  // filterTwoLowpass.input(HrSignal);
  // bandpassCurrent = filterOneHighpass.input(filterTwoLowpass.output());
   //Serial.println(bandpassCurrent);
   //Serial.print("Accelerometer Data:");
   //Serial.print(" ");
   //Serial.print(AccX);Serial.print(',');Serial.print(AccY);Serial.print(',');Serial.print(AccZ);Serial.print(',');Serial.print(GyroX);Serial.print(',');Serial.print(GyroY);Serial.print(',');Serial.print(GyroZ);Serial.print(',');
   //Serial.print("HRsensor raw data: ");
   //Serial.println(Signal);                    // Send the Signal value to Serial Plotter.
  // Serial.println(',');


   if(Signal > Threshold){                          // If the signal is above "550", then "turn-on" Arduino's on-Board LED.
     digitalWrite(LED13,HIGH);
   } else {
     digitalWrite(LED13,LOW);                //  Else, the sigal must be below "550", so "turn-off" this LED.
   }
   unsigned long CurrentTime = micros();
   count++;
   
   HrSignal = analogRead(PulseSensorPurplePin);  // Read the PulseSensor's value.

                                                  // Assign this value to the "Signal" variable.

   HrNoiseSignal = analogRead(MovementNoisePin); // Read the PulseSensor's value for the noise

                                                      // sensing pin. This senses no HR signals, just

                                                      // the interference from the noise

        

        filterTwoLowpass.input(HrSignal);             // Filter HR signal

        bandpassCurrent = filterOneHighpass.input(filterTwoLowpass.output());



        // filterTwoLpNoise.input(HrNoiseSignal);             // Filter Interference signal

        // bandpassNoiseCurrent = filterOneHpNoise.input(filterTwoLpNoise.output());



        // Save and update recent maximums

        if((TMinusOneHr > TMinusTwoHr)&&(TMinusOneHr > bandpassCurrent)){

            //Then we are at a peak

            if(bandpassCurrent > maxOne.value){

                maxThree = maxTwo;

                maxTwo = maxOne;

                maxOne.value = TMinusOneHr;

                maxOne.counter = counter;

            }

            else if(bandpassCurrent > maxTwo.value){

               maxThree = maxTwo;

               maxTwo.value = TMinusOneHr;

               maxTwo.counter = counter; 

            }

            else if(bandpassCurrent > maxThree.value){

              maxThree.value = TMinusOneHr;

              maxThree.counter = counter;

            }

        }

        
        
        // Clean old maximums 
        if(((counter - maxOne.counter)>maxValueTime)&&(maxOne.value != 0.00)){

            // Serial.print("\nMax one expired\n");

            maxOne = maxTwo;

            maxTwo = maxThree;

            maxThree.value = 0;

            maxThree.counter = counter;

        }

        if(((counter - maxTwo.counter)>maxValueTime)&&(maxTwo.value != 0.00)){

            // Serial.print("\nMax two expired\n");

            maxTwo = maxThree;

            maxThree.value = 0;

            maxThree.counter = counter;

        }

        if(((counter - maxThree.counter)>maxValueTime)&&(maxThree.value != 0.00)){

            //Serial.print("\nMax three expired\n");

            maxThree.value = 0;

            maxThree.counter = counter;

        }



        //Toggling LED with heart beat

        threshold = thresholdRatio*maxTwo.value;

        if((maxThree.value != 0.00)&&(bandpassCurrent >= threshold)){

            digitalWrite(13, HIGH);

            aboveThreshold = true;

        }

        else{

            digitalWrite(13, LOW);

            aboveThreshold = false;

        }



        //Find beats per minute

        if(aboveThreshold && (TMinusOneHr > TMinusTwoHr)&&(TMinusOneHr > bandpassCurrent)){

            if(beatCounter == 0.00){

              firstTime = millis();

              latestTime = millis();

            }

            else{

              lastTime = latestTime;

              latestTime = millis();

              instHr = 60.00*1000.00/(latestTime-lastTime);

              avgHr = beatCounter*60.00*1000.00/(latestTime-firstTime);

            }

            beatCounter += 1.00;

        }

        

       /* Serial.print(bandpassCurrent);

        Serial.print(",");

        //Serial.println(threshold);    

        Serial.print(instHr);

        Serial.print(",");

        Serial.print(avgHr);
        Serial.print(",");*/





        //Update variables for next run through

        TMinusTwoHr = TMinusOneHr;

        TMinusOneHr = bandpassCurrent;

        counter += 1;
      //  Serial.println("0");

        delay(20);
 }

   /*unsigned long ElapsedTime = CurrentTime - StartTime;
   unsigned long avg = (ElapsedTime+prev)/count;
   prev = prev + ElapsedTime;
   Serial.print("time for one loop is ");
   Serial.println(ElapsedTime);
   Serial.print("avg time for one loop is ");
   Serial.println(avg);*/


}

