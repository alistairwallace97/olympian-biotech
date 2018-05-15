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
            Serial.print("\nMax three expired\n");
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
        
        Serial.print(bandpassCurrent);
        Serial.print(",");
        Serial.println(threshold);    
        /*Serial.print(instHr);
        Serial.print(", ");
        Serial.println(avgHr);*/


        //Update variables for next run through
        TMinusTwoHr = TMinusOneHr;
        TMinusOneHr = bandpassCurrent;
        counter += 1;
        delay(10);
    }
}

void loop() {
    testHrFilter();
}


