#include <Filters.h>

//  Variables
int PulseSensorPurplePin = 0;        // Pulse Sensor PURPLE WIRE connected to ANALOG PIN 0
int LED13 = 13;   //  The on-board Arduion LED

float hrLowCutoff = 0.5;                      // 30 bpm
float hrHighCutoff = 4.166666;                // 250 bpm
float bandpassCurrent = 0;

void setup() {
  Serial.begin( 57600 );    // start the serial port
}

void testHrFilter () {
    // standard Lowpass, set to the corner frequency
    FilterTwoPole filterTwoLowpass;               // create a two pole Lowpass filter
    filterTwoLowpass.setAsFilter( LOWPASS_BUTTERWORTH, hrHighCutoff );
    FilterOnePole filterOneHighpass( HIGHPASS, hrLowCutoff );  // create a one pole (RC) highpass filter
    FilterTwoPole filterTwoLpNoise;               // create a two pole Lowpass filter
    filterTwoLpNoise.setAsFilter( LOWPASS_BUTTERWORTH, hrHighCutoff );
    FilterOnePole filterOneHpNoise( HIGHPASS, hrLowCutoff );  // create a one pole (RC) highpass filter
    int HrSignal;                                 // holds the incoming raw data. Signal value can range from 0-1024

    while(true) {     
        HrSignal = analogRead(PulseSensorPurplePin);  // Read the PulseSensor's value.
                                                  // Assign this value to the "Signal" variable.
        filterTwoLowpass.input(HrSignal);
        bandpassCurrent = filterOneHighpass.input(filterTwoLowpass.output());
        Serial.println(bandpassCurrent);

        delay(10);
    }
}

void loop() {
  testHrFilter();
}
