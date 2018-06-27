/*This code is designed for the olympian bio-tech vest and it collects sensor's data from pcb and stores them to the micro SD card during collection. After collection is stopped, it transmit the 
 * collected data saved on micro SD card to the phone app via bluethooth, after that it automatically deletes the stored data to make sure micro SD card always has availale storage space for collection.
 */
#include <SPI.h>
#include <SD.h>
#include <Arduino.h>
#include "Adafruit_BLE.h"
#include "Adafruit_BluefruitLE_SPI.h"
#include "Adafruit_BluefruitLE_UART.h"
#include <Wire.h>
#include <Filters.h>
#include "BluefruitConfig.h"

#if SOFTWARE_SERIAL_AVAILABLE
#include <SoftwareSerial.h>
#endif

/*-----------------------------------------------------------------------*/
#define FACTORYRESET_ENABLE 0
#define MINIMUM_FIRMWARE_VERSION "0.6.6"
#define MODE_LED_BEHAVIOUR "MODE"
/*=========================================================================*/

//  Variables
int HR1Pin = A0; // Pulse Sensor PURPLE WIRE connected to ANALOG PIN 0
int HR2Pin = A1;
int LED13 = 13; //  The on-board Arduion LED
const int PIEZO_PIN1 = A4; // Piezo output
const int PIEZO_PIN2 = A5; // Piezo output
const int EMG_PIN1 = A2;
const int EMG_PIN2 = A3;
const int SW_PIN = 5;
const int chipSelect = 9;//CS pin for SD card

int linecounter; //to count how many lines of data are stored in sd card
int buttonstate = 0;
bool collection = false;
bool transmission = false;

String buffer;

Adafruit_BluefruitLE_SPI ble(BLUEFRUIT_SPI_CS, BLUEFRUIT_SPI_IRQ, BLUEFRUIT_SPI_RST);

// for debugging purpose
void error(const __FlashStringHelper* err)
{
    Serial.println(err);
    while (1);
}

const int MPU6050_addr = 0x68; // holds the incoming raw data. Signal value can range from 0-1024
int16_t AccX, AccY, AccZ, Temp, GyroX, GyroY, GyroZ, AccX_init, AccY_init, AccZ_init, Temp_init, GyroX_init, GyroY_init, GyroZ_init;
int count = 0;
int HR1; // holds the incoming raw data. Signal value can range from 0-1024
int HR2; // holds the incoming raw data. Signal value can range from 0-1024

File myFile;

void setup()
{
    /*while (!Serial);  // for debugging purpose
    delay(500);*/
    Wire.begin();
    Wire.beginTransmission(MPU6050_addr);
    Wire.write(0x6B);
    Wire.write(0);
    Wire.endTransmission(true);
    pinMode(LED13, OUTPUT);
    pinMode(SW_PIN, INPUT);
    Serial.begin(9600); // Set's up Serial Communication at certain speed.

    if (!ble.begin(VERBOSE_MODE)) {
        error(F("Couldn't find Bluefruit, make sure it's in CoMmanD mode & check wiring?"));
    }
    ble.echo(false);
    ble.verbose(false);
    
    if (!SD.begin(9)) {
        Serial.println("initialization failed!");
        return;
    }
    Serial.println("initialization done.");

    // open the file. note that only one file can be open at a time,
    // so you have to close this one before opening another.
    myFile = SD.open("data.txt", FILE_WRITE);

    // if the file opened okay, write to it:
    if (myFile) {
        Serial.println("testing sdcard!");
        // close the file:
        myFile.close();
        Serial.println("done.");
    }
}

void loop()
{
    //------check for the 'start collection' signal from phone app------
    linecounter = 0;//counter to counter number of lines stored in datafile, used for the progress bar during transmission
    if (ble.isConnected()) {
        Serial.println("bluetooth connected!");
        ble.println("AT+BLEUARTRX");
        ble.readline();
        // Serial.println(ble.buffer);
        if (strcmp(ble.buffer, "START") == 0) {
            collection = true;
        }
        ble.waitForOK();
    }
    while (collection) {
        linecounter = linecounter + 1;
        myFile = SD.open("data.txt", FILE_WRITE);
        //Serial.println("into collection mode");
        if (myFile) {
            Serial.println("Start collection to sd card!");
            //---------------acccelerometer setup----------------
            unsigned long StartTime = micros();
            Wire.beginTransmission(MPU6050_addr);
            Wire.write(0x3B);
            Wire.endTransmission(false);
            Wire.requestFrom(MPU6050_addr, 14, true);

            //-----------Buttons------------------
            buttonstate = digitalRead(SW_PIN);
            if (buttonstate == LOW) {
                //myFile.print("cough state: ");
                myFile.print("1");
                myFile.print(",");
            }
            else {
                //myFile.print("cough state: ");
                myFile.print("0");
                myFile.print(",");
            }

            //------------EMG sensor------------------
            int EMGADC1 = analogRead(EMG_PIN1);
            int EMGADC2 = analogRead(EMG_PIN2);
            //myFile.print("EMGdata: ");
            myFile.print(EMGADC1);
            myFile.print(",");
            myFile.print(EMGADC2);
            myFile.print(",");
            //----------------------------------------

            //----------------vibration sensor----------------------------
            // Read Piezo ADC value in, and convert it to a voltage
            int piezoADC1 = analogRead(PIEZO_PIN1);
            float piezoV1 = piezoADC1 / 1023.0 * 100;
            myFile.print(piezoV1); // Print the voltage.
            myFile.print(",");
            int piezoADC2 = analogRead(PIEZO_PIN2);
            float piezoV2 = piezoADC2 / 1023.0 * 100;
            myFile.print(piezoV2); // Print the voltage.
            myFile.print(",");
            //----------------------------------------------------------------

            //---------accelerometer sensor-------------------------------------
            AccX = Wire.read() << 8 | Wire.read();
            AccY = Wire.read() << 8 | Wire.read();
            AccZ = Wire.read() << 8 | Wire.read();
            //Temp=Wire.read()<<8|Wire.read();
            GyroX = Wire.read() << 8 | Wire.read();
            GyroY = Wire.read() << 8 | Wire.read();
            GyroZ = Wire.read() << 8 | Wire.read();
            //myFile.print("Accelerometer Data:");
            //myFile.print(" ");
            myFile.print(AccX);
            myFile.print(',');
            myFile.print(AccY);
            myFile.print(',');
            myFile.print(AccZ);
            myFile.print(',');
            myFile.print(GyroX);
            myFile.print(',');
            myFile.print(GyroY);
            myFile.print(',');
            myFile.print(GyroZ);
            myFile.print(',');
            //--------------------------------------------------------------------
            HR1 = analogRead(HR1Pin); // Read the PulseSensor's value.
            HR2 = analogRead(HR2Pin); //
            myFile.print(HR1);
            myFile.print(",");
            myFile.print(HR2);
            myFile.print(",");
            myFile.print("0");
            myFile.print(",");

            //Index for test person, change to your index accordinglybefore testing------------------------
            myFile.print("6"); //0: Alistair 1:Sugi 2:Lu 3:Ray 4:Ian 5:Ismaeel 6:Shamim 7:Others
            myFile.println("/"); //line ending for the phone app to decode

            //code for stop data collection and start transmission
            if (ble.isConnected()) {
                ble.println("AT+BLEUARTRX");
                ble.readline();
                //if received "STOP" then jump out of the collection loop and start transmission
                if (strcmp(ble.buffer, "STOP") == 0) { 
                    collection = false;
                    transmission = true;
                }
            }
            myFile.close();
        } 
        else {
            // if the file didn't open, print an error:
            ble.println("error opening data.txt");
        }

        delay(20);
    }
    //get out of the while loop and start transmission
    if (transmission) {
        ble.print("AT+BLEUARTTX=");
        ble.print("S");// Start transmission symbol for phone app
        ble.print(linecounter);//how many lines of data stored on sd card in total
        ble.println("/");
        ble.waitForOK();
        Serial.println("start transmission!");
        myFile = SD.open("data.txt");
        if (myFile) {
            Serial.println("file opened");
            // read from the file until there's nothing else in it:
            while (myFile.available()) {
                //if the bluethooth connection is disconnected during transmission, pause the transmission for reconnection
                while (!ble.isConnected()) {
                  delay(100);//wait for reconnection
                }
                //read the stored data line by line and save each line into a buffer, then transmit the content in buffer via bluebooth
                buffer = myFile.readStringUntil('\n');
                ble.print("AT+BLEUARTTX=");
                ble.println(buffer);
                ble.waitForOK();
            }
            // close the file:
            myFile.close();
        }
        else {
            // if the file didn't open, print an error:
            ble.println("error opening data file");
        }
        ble.print("AT+BLEUARTTX=");
        ble.println("D"); //character to indentify the end of transmission
        ble.waitForOK();
        transmission = false; //transmission done
        SD.remove("data.txt"); //remove the data file to save storage space on sd card
    }
}
