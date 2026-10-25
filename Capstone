/ * Voice-Controlled LED System using ESP32 DevKit V1
 * 19-class model → grouped into 8 logical commands
 * Compatible with ESP32 Arduino Core v3.x
 */

#include <ei.h>
#include <driver/i2s.h>

// ==========================================
// 1. PIN DEFINITIONS
// ==========================================
#define RED_LED_PIN     18
#define YELLOW_LED_PIN  19
#define PURPLE_LED_PIN  23
#define WHITE_LED_PIN   22

#define PWM_FREQ        5000
#define PWM_RESOLUTION  8

// PWM brightness levels (0-255)
#define BRIGHT_100  255
#define BRIGHT_50   128
#define BRIGHT_20   51
#define BRIGHT_OFF  0

// ==========================================
// 2. AUDIO CONFIGURATION (INMP441)
// ==========================================
#define I2S_PORT            I2S_NUM_0
#define SAMPLE_RATE         16000
#define SLICE_SIZE          4000

static float    inference_buffer[SLICE_SIZE];
static int16_t  temp_buffer[SLICE_SIZE];

// ==========================================
// 3. THRESHOLDS (per variant)
// ==========================================
#define TH_POWER_ON       0.70f
#define TH_SHUT_DOWN      0.60f
#define TH_RED            0.70f
#define TH_RED_HIGH       0.70f
#define TH_RED_LOW        0.60f
#define TH_RED_MEDIUM     0.75f
#define TH_YELLOW         0.85f
#define TH_YELLOW_HIGH    0.85f
#define TH_YELLOW_LOW     0.50f
#define TH_YELLOW_MEDIUM  0.75f
#define TH_PURPLE         0.85f
#define TH_PURPLE_HIGH    0.85f
#define TH_PURPLE_LOW     0.60f
#define TH_PURPLE_MEDIUM  0.75f
#define TH_WHITE          0.80f
#define TH_WHITE_HIGH     0.80f
#define TH_WHITE_LOW      0.55f
#define TH_WHITE_MEDIUM   0.75f
#define TH_NOISE          0.90f
#define TH_DEFAULT        0.70f

// ==========================================
// 4. FUNCTION PROTOTYPES
// ==========================================
void setAllLeds(int brightness);
void updateSpecificLed(int activePin, int brightness);
void processInferenceResults(ei_impulse_result_t result);
void handleCommand(String command, int brightness);
void runContinuousInference();
float thresholdForLabel(String label);
String normalizeLabel(String label);
int   brightnessForLabel(String rawLabel);

// ==========================================
// 5. SETUP
// ==========================================
void setup() {
  Serial.begin(115200);
  delay(1000);
  Serial.println("Starting Voice-Controlled LEDs on DevKit V1...");

  ledcAttach(RED_LED_PIN,    PWM_FREQ, PWM_RESOLUTION);
  ledcAttach(YELLOW_LED_PIN, PWM_FREQ, PWM_RESOLUTION);
  ledcAttach(PURPLE_LED_PIN, PWM_FREQ, PWM_RESOLUTION);
  ledcAttach(WHITE_LED_PIN,  PWM_FREQ, PWM_RESOLUTION);

  setAllLeds(BRIGHT_OFF);

  i2s_config_t i2s_config = {
    .mode = (i2s_mode_t)(I2S_MODE_MASTER | I2S_MODE_RX),
    .sample_rate = SAMPLE_RATE,
    .bits_per_sample = I2S_BITS_PER_SAMPLE_16BIT,
    .channel_format = I2S_CHANNEL_FMT_ONLY_LEFT,
    .communication_format = I2S_COMM_FORMAT_STAND_I2S,
    .intr_alloc_flags = ESP_INTR_FLAG_LEVEL1,
    .dma_buf_count = 8,
    .dma_buf_len = 512,
    .use_apll = false,
    .tx_desc_auto_clear = false,
    .fixed_mclk = 0
  };

  i2s_pin_config_t pin_config = {
    .bck_io_num   = 26,
    .ws_io_num    = 25,
    .data_out_num = I2S_PIN_NO_CHANGE,
    .data_in_num  = 33
  };

  i2s_driver_install(I2S_PORT, &i2s_config, 0, NULL);
  i2s_set_pin(I2S_PORT, &pin_config);

  run_classifier_init();

  Serial.println("Ready. Speak a command...");
}

// ==========================================
// 6. MAIN LOOP
// ==========================================
void loop() {
  runContinuousInference();
}

// ==========================================
// 7. CONTINUOUS INFERENCE
// ==========================================
void runContinuousInference() {
  size_t bytes_read = 0;
  i2s_read(I2S_PORT, (void*)temp_buffer, SLICE_SIZE * sizeof(int16_t), &bytes_read, portMAX_DELAY);

  for (int i = 0; i < SLICE_SIZE; i++) {
    inference_buffer[i] = (float)temp_buffer[i];
  }

  signal_t signal;
  int err = numpy::signal_from_buffer(inference_buffer, SLICE_SIZE, &signal);
  if (err != 0) {
    Serial.printf("Signal error: %d\n", err);
    return;
  }

  ei_impulse_result_t result = { 0 };
  err = run_classifier_continuous(&signal, &result, false, true);
  if (err != EI_IMPULSE_OK) {
    Serial.printf("Classifier error: %d\n", err);
    return;
  }

  processInferenceResults(result);
}

// ==========================================
// 8. THRESHOLD LOOKUP
// ==========================================
float thresholdForLabel(String label) {
  if (label == "Power on")      return TH_POWER_ON;
  if (label == "Shut down")     return TH_SHUT_DOWN;
  if (label == "red")           return TH_RED;
  if (label == "red high")      return TH_RED_HIGH;
  if (label == "red low")       return TH_RED_LOW;
  if (label == "red medium")    return TH_RED_MEDIUM;
  if (label == "yellow")        return TH_YELLOW;
  if (label == "yellow high")   return TH_YELLOW_HIGH;
  if (label == "Yellow high")   return TH_YELLOW_HIGH;
  if (label == "yellow low")    return TH_YELLOW_LOW;
  if (label == "yellow medium") return TH_YELLOW_MEDIUM;
  if (label == "purple")        return TH_PURPLE;
  if (label == "Purple high")   return TH_PURPLE_HIGH;
  if (label == "Purple low")    return TH_PURPLE_LOW;
  if (label == "Purple medium") return TH_PURPLE_MEDIUM;
  if (label == "white")         return TH_WHITE;
  if (label == "white high")    return TH_WHITE_HIGH;
  if (label == "White low")     return TH_WHITE_LOW;
  if (label == "white medium")  return TH_WHITE_MEDIUM;
  if (label == "noise")         return TH_NOISE;
  return TH_DEFAULT;
}

// ==========================================
// 9. GROUP LABELS INTO PARENT COMMANDS
// ==========================================
String normalizeLabel(String label) {
  if (label == "Power on")                              return "Power on";
  if (label == "Shut down")                             return "Shut down";

  if (label == "red" || label == "red high" ||
      label == "red low" || label == "red medium")      return "red";

  if (label == "yellow" || label == "yellow high" ||
      label == "Yellow high" ||
      label == "yellow low" || label == "yellow medium") return "yellow";

  if (label == "purple" || label == "Purple high" ||
      label == "Purple low" || label == "Purple medium") return "purple";

  if (label == "white" || label == "white high" ||
      label == "White low" || label == "white medium")   return "white";

  return "";  // noise or unknown
}

// ==========================================
// 10. BRIGHTNESS LOOKUP (the key change!)
// ==========================================
// "red"          → 100%
// "red high"     → 100%
// "red medium"   → 50%
// "red low"      → 20%
// Same rule applied to yellow, purple, white
int brightnessForLabel(String rawLabel) {
  // Off / all-on handled separately
  if (rawLabel == "Shut down") return BRIGHT_OFF;

  // Intensity-specific labels
  if (rawLabel.endsWith("high"))   return BRIGHT_100;
  if (rawLabel.endsWith("medium")) return BRIGHT_50;
  if (rawLabel.endsWith("low"))    return BRIGHT_20;

  // Bare color name (no variant) → full brightness
  return BRIGHT_100;
}

// ==========================================
// 11. LOGIC
// ==========================================
void processInferenceResults(ei_impulse_result_t result) {
  float max_confidence = 0.0;
  int max_index = -1;

  for (size_t ix = 0; ix < EI_CLASSIFIER_LABEL_COUNT; ix++) {
    if (result.classification[ix].value > max_confidence) {
      max_confidence = result.classification[ix].value;
      max_index = ix;
    }
  }

  if (max_index < 0) return;

  String rawLabel  = String(result.classification[max_index].label);
  String command   = normalizeLabel(rawLabel);
  float  threshold = thresholdForLabel(rawLabel);
  int    brightness = brightnessForLabel(rawLabel);

  // Debug output (comment out once tuned)
  Serial.printf("%-15s -> %-10s : %.2f (need %.2f) | PWM %d\n",
                rawLabel.c_str(), command.c_str(),
                max_confidence, threshold, brightness);

  if (command.length() == 0) return;   // noise / unknown

  if (max_confidence >= threshold) {
    handleCommand(command, brightness);
  }
}

void handleCommand(String command, int brightness) {
  if (command == "Power on") {
    setAllLeds(BRIGHT_100);
    Serial.println(">>> ALL LEDs ON (100%)");
  }
  else if (command == "Shut down") {
    setAllLeds(BRIGHT_OFF);
    Serial.println(">>> ALL LEDs OFF");
  }
  else if (command == "red") {
    updateSpecificLed(RED_LED_PIN, brightness);
    Serial.printf(">>> RED at %d\n", brightness);
  }
  else if (command == "yellow") {
    updateSpecificLed(YELLOW_LED_PIN, brightness);
    Serial.printf(">>> YELLOW at %d\n", brightness);
  }
  else if (command == "purple") {
    updateSpecificLed(PURPLE_LED_PIN, brightness);
    Serial.printf(">>> PURPLE at %d\n", brightness);
  }
  else if (command == "white") {
    updateSpecificLed(WHITE_LED_PIN, brightness);
    Serial.printf(">>> WHITE at %d\n", brightness);
  }
}

// ==========================================
// 12. HELPERS
// ==========================================
void setAllLeds(int brightness) {
  ledcWrite(RED_LED_PIN,    brightness);
  ledcWrite(YELLOW_LED_PIN, brightness);
  ledcWrite(PURPLE_LED_PIN, brightness);
  ledcWrite(WHITE_LED_PIN,  brightness);
}

void updateSpecificLed(int activePin, int brightness) {
  ledcWrite(RED_LED_PIN,    (activePin == RED_LED_PIN)    ? brightness : 0);
  ledcWrite(YELLOW_LED_PIN, (activePin == YELLOW_LED_PIN) ? brightness : 0);
  ledcWrite(PURPLE_LED_PIN, (activePin == PURPLE_LED_PIN) ? brightness : 0);
  ledcWrite(WHITE_LED_PIN,  (activePin == WHITE_LED_PIN)  ? brightness : 0);
}
