const int SW_PIN = 5;
int buttonstate = 0;

void setup() {
  // put your setup code here, to run once:
  pinMode(SW_PIN,INPUT);
  Serial.begin(9600);
  
}

void loop() {
  buttonstate = digitalRead(SW_PIN);
  /*if (buttonstate){
    Serial.print("cough state: ");
    Serial.println("1");
  }
  else {
    Serial.print("cough state: ");
    Serial.println("0");
  }*/
  Serial.println(buttonstate);
}
