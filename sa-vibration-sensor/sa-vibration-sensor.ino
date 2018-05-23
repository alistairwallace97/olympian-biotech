const int PIEZO_PIN1 = A0; // Piezo output
const int PIEZO_PIN2 = A1; // Piezo output


void setup() 
{
  Serial.begin(9600);
}

void loop() 
{
  // Read Piezo ADC value in, and convert it to a voltage
  int piezoADC1 = analogRead(PIEZO_PIN1);
  float piezoV1 = piezoADC1 / 1023.0 * 100;
  int piezoADC2 = analogRead(PIEZO_PIN2);
  float piezoV2 = piezoADC2 / 1023.0 * 100;
  Serial.print(piezoV1); // Print the voltage.
  Serial.print(", ")
  Serial.println(piezoV2); // Print the voltage.
}
