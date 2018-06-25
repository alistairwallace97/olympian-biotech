
/*  PulseSensor™ Starter Project and Signal Tester
 *  The Best Way to Get Started  With, or See the Raw Signal of, your PulseSensor™ & Arduino.
 *
 *  Here is a link to the tutorial
 *  https://pulsesensor.com/pages/code-and-guide
 *
 *  WATCH ME (Tutorial Video):
 *  https://www.youtube.com/watch?v=82T_zBZQkOE
 *
 *
-------------------------------------------------------------
1) This shows a live human Heartbeat Pulse.
2) Live visualization in Arduino's Cool "Serial Plotter".
3) Blink an LED on each Heartbeat.
4) This is the direct Pulse Sensor's Signal.
5) A great first-step in troubleshooting your circuit and connections.
6) "Human-readable" code that is newbie friendly."

*/
#include <Arduino.h>
#include <SPI.h>
#include "Adafruit_BLE.h"
#include "Adafruit_BluefruitLE_SPI.h"
#include "Adafruit_BluefruitLE_UART.h"
#include<Wire.h>


//  Variables
int PulseSensorPurplePin = 0;        // Pulse Sensor PURPLE WIRE connected to ANALOG PIN 0
int LED13 = 13;   //  The on-board Arduion LED

const int MPU6050_addr=0x68; // holds the incoming raw data. Signal value can range from 0-1024
int16_t AccX,AccY,AccZ,Temp,GyroX,GyroY,GyroZ,AccX_init,AccY_init,AccZ_init,Temp_init,GyroX_init,GyroY_init,GyroZ_init;
bool initalize=1;
int16_t Signal;
int Threshold = 550;            // Determine which Signal to "count as a beat", and which to ingore.
int count = 0;
unsigned long prev = 0;

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

  AccX=Wire.read()<<8|Wire.read();
  AccY=Wire.read()<<8|Wire.read();
  AccZ=Wire.read()<<8|Wire.read();
  //Temp=Wire.read()<<8|Wire.read();
  GyroX=Wire.read()<<8|Wire.read();
  GyroY=Wire.read()<<8|Wire.read();
  GyroZ=Wire.read()<<8|Wire.read();

  Signal = analogRead(PulseSensorPurplePin);  // Read the PulseSensor's value.
                                              // Assign this value to the "Signal" variable.
  // Serial.println("Accelerometer Data:");
  // Serial.print(AccX);Serial.print(',');Serial.print(AccY);Serial.print(',');Serial.print(AccZ);Serial.print(',');Serial.print(GyroX);Serial.print(',');Serial.print(GyroY);Serial.print(',');Serial.println(GyroZ);
 //  Serial.println("HRsensor raw data");
   Serial.println(Signal);                    // Send the Signal value to Serial Plotter.


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
  // Serial.print("time for one loop is ");
   //Serial.println(ElapsedTime);
   //Serial.print("avg time for one loop is ");
   //Serial.println(avg);


}
