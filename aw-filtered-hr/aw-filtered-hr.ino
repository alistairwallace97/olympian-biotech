#include <Filters.h>

//  Variables
int PulseSensorPurplePin = 1;         // Pulse Sensor PURPLE WIRE connected to ANALOG PIN 0
int MovementNoisePin = 0;             // Pulse Sensor PURPLE WIRE connected to ANALOG PIN 1
int LED13 = 13;   //  The on-board Arduino LED


void setup() {
  Serial.begin( 57600 );    // start the serial port
}

void testHrFilter () {
    //Initialisation 
    float hrLowCutoff = 0.5;                      // 30 bpm
    float hrHighCutoff = 4.166666;                // 250 bpm
    float bandpassCurrent = 0;    
    float bandpassNoiseCurrent = 0; 

    // standard Lowpass, set to the corner frequency
    FilterTwoPole filterTwoLowpass;               // create a two pole Lowpass filter
    filterTwoLowpass.setAsFilter( LOWPASS_BUTTERWORTH, hrHighCutoff );
    FilterOnePole filterOneHighpass( HIGHPASS, hrLowCutoff );  // create a one pole (RC) highpass filter
    
    FilterTwoPole filterTwoLpNoise;               // create a two pole Lowpass filter
    filterTwoLpNoise.setAsFilter( LOWPASS_BUTTERWORTH, hrHighCutoff );
    FilterOnePole filterOneHpNoise( HIGHPASS, hrLowCutoff );  // create a one pole (RC) highpass filter
    
    int HrSignal;                                 // holds the incoming raw data. Signal value can range from 0-1024
    int HrNoiseSignal;                                 // holds the incoming raw data. Signal value can range from 0-1024

    while(true) {     
        HrSignal = analogRead(PulseSensorPurplePin);  // Read the PulseSensor's value.
                                                  // Assign this value to the "Signal" variable.
        HrNoiseSignal = analogRead(MovementNoisePin); // Read the PulseSensor's value for the noise
                                                      // sensing pin. This senses no HR signals, just
                                                      // the interference from the noise
        
        filterTwoLowpass.input(HrSignal);             // Filter HR signal
        bandpassCurrent = filterOneHighpass.input(filterTwoLowpass.output());

        filterTwoLpNoise.input(HrNoiseSignal);             // Filter Interference signal
        bandpassNoiseCurrent = filterOneHpNoise.input(filterTwoLpNoise.output());

        Serial.print(bandpassCurrent);
        Serial.print(",");
        Serial.println(bandpassNoiseCurrent);
        //Serial.print(",");
        //Serial.println(bandpassCurrent + bandpassNoiseCurrent);

        delay(10);
    }
}

void loop() {
    testHrFilter();
}


// Not currently working or used
float bandpassFilter(float Signal) {
    //Initialisation 
    float hrLowCutoff = 0.5;                      // 30 bpm
    float hrHighCutoff = 4.166666;                // 250 bpm
    float bandpassCurrent = 0;
    
    // standard Lowpass, set to the corner frequency
    FilterTwoPole filterTwoLowpass;               // create a two pole Lowpass filter
    filterTwoLowpass.setAsFilter( LOWPASS_BUTTERWORTH, hrHighCutoff );
    FilterOnePole filterOneHighpass( HIGHPASS, hrLowCutoff );  // create a one pole (RC) highpass filter

    filterTwoLowpass.input(Signal);             // Filter HR signal
    bandpassCurrent = filterOneHighpass.input(filterTwoLowpass.output());

    Serial.print("HrSignal in funtion = "); 
    Serial.print(Signal);
    Serial.print(", ");   
    Serial.print("bandpassCurrent in funtion = ");
    Serial.print(bandpassCurrent);
    Serial.print(", ");

    return bandpassCurrent;
}

