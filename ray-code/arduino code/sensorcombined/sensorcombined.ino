
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
int LED13 = 13;   //  The on-board Arduion LED
const int PIEZO_PIN = 1; // Piezo output

float hrLowCutoff = 0.5;                      // 30 bpm
float hrHighCutoff = 4.166666;                // 250 bpm
float bandpassCurrent = 0;




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

// The SetUp Function:
void setup() {
    Wire.begin();
    Wire.beginTransmission(MPU6050_addr);
    Wire.write(0x6B);
    Wire.write(0);
    Wire.endTransmission(true);
    pinMode(LED13,OUTPUT);         // pin that will blink to your heartbeat!
    Serial.begin(9600);         // Set's up Serial Communication at certain speed.
}

// The Main Loop Function
void loop() {
  unsigned long StartTime = micros();
  Wire.beginTransmission(MPU6050_addr);
  Wire.write(0x3B);
  Wire.endTransmission(false);
  Wire.requestFrom(MPU6050_addr,14,true);

  //----------------vibration sensor----------------------------
  // Read Piezo ADC value in, and convert it to a voltage
  int piezoADC = analogRead(PIEZO_PIN);
  float piezoV = piezoADC / 1023.0 * 100;
  //Serial.println(piezoV); // Print the voltage.
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
   Serial.println("Accelerometer Data:");
   Serial.print(AccX);Serial.print(',');Serial.print(AccY);Serial.print(',');Serial.print(AccZ);Serial.print(',');Serial.print(GyroX);Serial.print(',');Serial.print(GyroY);Serial.print(',');Serial.println(GyroZ);
   Serial.println("HRsensor raw data");
   Serial.println(Signal);                    // Send the Signal value to Serial Plotter.
   //Serial.print(',');


   if(Signal > Threshold){                          // If the signal is above "550", then "turn-on" Arduino's on-Board LED.
     digitalWrite(LED13,HIGH);
   } else {
     digitalWrite(LED13,LOW);                //  Else, the sigal must be below "550", so "turn-off" this LED.
   }
   unsigned long CurrentTime = micros();
   count++;
   delay(10);
   unsigned long ElapsedTime = CurrentTime - StartTime;
   unsigned long avg = (ElapsedTime+prev)/count;
   prev = prev + ElapsedTime;
   //Serial.print("time for one loop is ");
   //Serial.println(ElapsedTime);
   //Serial.print("avg time for one loop is ");
   //Serial.println(avg);


}

/*void testHrFilter () {
    // standard Lowpass, set to the corner frequency
    FilterTwoPole filterTwoLowpass;               // create a two pole Lowpass filter
    filterTwoLowpass.setAsFilter( LOWPASS_BUTTERWORTH, hrHighCutoff );
    FilterOnePole filterOneHighpass( HIGHPASS, hrLowCutoff );  // create a one pole (RC) highpass filter
    int HrSignal;                               // holds the incoming raw data. Signal value can range from 0-1024
    
        HrSignal = analogRead(PulseSensorPurplePin);  // Read the PulseSensor's value.
                                                  // Assign this value to the "Signal" variable.
        filterTwoLowpass.input(HrSignal);
        bandpassCurrent = filterOneHighpass.input(filterTwoLowpass.output());
        Serial.println(bandpassCurrent);

}*/
