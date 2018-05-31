#include <Filters.h>

//  Variables
int PulseSensorPurplePin = 0;         // Pulse Sensor PURPLE WIRE connected to ANALOG PIN 0
int MovementNoisePin = 1;             // Pulse Sensor PURPLE WIRE connected to ANALOG PIN 1
int LED13 = 13;   //  The on-board Arduino LED

struct recentMax {
  float value;
  int counter;
};

void setup() {
  Serial.begin( 57600 );    // start the serial port
}

void testHrFilter () {
    //Initialisation 
    float hrLowCutoff = 0.5;                      // 30 bpm
    float hrHighCutoff = 4.166666;                // 250 bpm
    float bandpassCurrent = 0;    
    float bandpassNoiseCurrent = 0; 
    float TMinusOneHr = 0;
    float TMinusTwoHr = 0;

    // standard Lowpass, set to the corner frequency
    FilterTwoPole filterTwoLowpass;               // create a two pole Lowpass filter
    filterTwoLowpass.setAsFilter( LOWPASS_BUTTERWORTH, hrHighCutoff );
    FilterOnePole filterOneHighpass( HIGHPASS, hrLowCutoff );  // create a one pole (RC) highpass filter
    
    FilterTwoPole filterTwoLpNoise;               // create a two pole Lowpass filter
    filterTwoLpNoise.setAsFilter( LOWPASS_BUTTERWORTH, hrHighCutoff );
    FilterOnePole filterOneHpNoise( HIGHPASS, hrLowCutoff );  // create a one pole (RC) highpass filter
    
    int HrSignal;                                 // holds the incoming raw data. Signal value can range from 0-1024
    int HrNoiseSignal;                                 // holds the incoming raw data. Signal value can range from 0-1024

    int counter = 0;
    recentMax maxOne;
    maxOne.value = 0;
    maxOne.counter = 0;
    recentMax maxTwo;
    maxTwo.value = 0;
    maxTwo.counter = 0;
    recentMax maxThree;
    maxThree.value = 0;
    maxThree.counter = 0;

    while(true) {     
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
            // Serial.println("\n\n\n\nFound a maximum");
            // Serial.print("maxOne.value = ");
            // Serial.print(maxOne.value);
            // Serial.print(", maxOne.counter = ");
            // Serial.println(maxOne.counter);
            // Serial.print("maxTwo.value = ");
            // Serial.print(maxTwo.value);
            // Serial.print(", maxTwo.counter = ");
            // Serial.println(maxTwo.counter);
            // Serial.print("maxThree.value = ");
            // Serial.print(maxThree.value);
            // Serial.print(", maxThree.counter = ");
            // Serial.println(maxThree.counter);
            // Serial.print("bandpassCurrent = ");
            // Serial.println(bandpassCurrent);
            // Serial.println("\n\n\n");
            //Then we are at a peak
            if(HrSignal > maxOne.value){
                maxThree = maxTwo;
                maxTwo = maxOne;
                maxOne.value = TMinusOneHr;
                maxOne.counter = counter;
            }
            else if(HrSignal > maxTwo.value){
               maxThree = maxTwo;
               maxTwo.value = TMinusOneHr;
               maxTwo.counter = counter; 
            }
            else if(HrSignal > maxThree.value){
              maxThree.value = TMinusOneHr;
              maxThree.counter = counter;
            }
        }
        
        // Clean old maximums 
        if(((counter - maxOne.counter)>60)&&(maxOne.value != 0.00)){
            // Serial.print("\nMax one expired\n");
            maxOne = maxTwo;
            maxTwo = maxThree;
            maxThree.value = 0;
            maxThree.counter = counter;
        }
        if(((counter - maxTwo.counter)>60)&&(maxTwo.value != 0.00)){
            // Serial.print("\nMax two expired\n");
            maxTwo = maxThree;
            maxThree.value = 0;
            maxThree.counter = counter;
        }
        if(((counter - maxThree.counter)>60)&&(maxThree.value != 0.00)){
            // Serial.print("\nMax three expired\n");
            maxThree.value = 0;
            maxThree.counter = counter;
        }
        
        Serial.print(HrSignal);
        Serial.print("     ");
        Serial.println(bandpassCurrent);
        // Serial.print(",");
        // Serial.print(TMinusOneHr);
        // Serial.print(",");
        // Serial.print(TMinusTwoHr);
        // Serial.print(",   ");
        // Serial.print(maxOne.value);
        // Serial.print(",");
        // Serial.print(maxTwo.value);
        // Serial.print(",");
        // Serial.println(maxThree.value);
        // Serial.println(bandpassNoiseCurrent);
        // Serial.print(",");
        // Serial.println(bandpassCurrent + bandpassNoiseCurrent);

        TMinusTwoHr = TMinusOneHr;
        TMinusOneHr = bandpassCurrent;
        counter += 1;
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

