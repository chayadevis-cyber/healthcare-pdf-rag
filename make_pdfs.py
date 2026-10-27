from reportlab.lib.pagesizes import A4
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, PageBreak
from reportlab.lib.styles import getSampleStyleSheet
styles = getSampleStyleSheet()
NOTE = "Sample document for a course project. CityCare Hospital is fictional; all policies, times and amounts are invented for demonstration and are not medical advice."

DOCS = {
"patient_admission_discharge_guide.pdf": ("CityCare Hospital - Patient Admission and Discharge Guide", [
 ("1. Admission Process", [
  "The admission desk at CityCare Hospital is open 24 hours a day. For planned admissions, patients should arrive at least one hour before the scheduled time so that registration and bed allotment can be completed.",
  "For every admission, please bring a government-issued photo ID, your health insurance card, the doctor's referral or admission letter, and all previous medical reports related to the current condition."]),
 ("2. Visiting Hours", [
  "General ward visiting hours are 11 AM to 1 PM and 5 PM to 7 PM. A maximum of two visitors are allowed per patient at a time, and children under 12 years are not permitted inside the wards.",
  "In the intensive care unit, visits are limited to 15 minutes, twice a day. Visitors must sanitise their hands at the entrance and follow the instructions of the nursing staff."]),
 ("3. Patient Rights and Responsibilities", [
  "Every patient has the right to be treated with respect, to have their medical information kept confidential, to understand their diagnosis and treatment options, and to give or refuse informed consent before any procedure. Patients may also request a second opinion at any time.",
  "Patients are expected to provide accurate medical history, follow the care plan agreed with the treating team, and respect hospital rules and other patients."]),
 ("4. Raising a Concern", [
  "Patients and families can submit complaints through the patient relations desk, the hospital app or the feedback boxes on each floor. The hospital acknowledges every complaint within 48 hours and shares the outcome of its review within 7 working days."]),
 ("5. Discharge Process", [
  "Discharge is planned by the treating doctor. Once the doctor approves discharge, the nursing team prepares the discharge summary within 3 hours. Patients are expected to vacate the room before 11 AM on the day of discharge.",
  "The discharge summary lists the diagnosis, procedures performed, medicines to continue at home, warning signs, and the date of the follow-up visit. Please check that all information is correct before leaving."]),
 ("6. After Discharge", [
  "A follow-up appointment is normally scheduled within 7 days of discharge. If you notice warning signs listed in your discharge summary, contact the hospital emergency line or visit the emergency department immediately.",
  "A copy of your medical records can be requested from the medical records department with a photo ID. Records are issued within 3 working days for a fee of 200 rupees."]),
]),
"insurance_billing_faq.pdf": ("CityCare Hospital - Insurance and Billing FAQ for Patients", [
 ("1. Cashless Treatment", [
  "CityCare Hospital works with most major insurance companies and third-party administrators to offer cashless treatment. To use it, show your insurance card and photo ID at the insurance desk.",
  "For planned admissions, the insurance desk submits a pre-authorisation request to the insurer. Approval normally takes 4 hours. In emergencies, treatment starts immediately and the request is submitted within 2 hours of admission."]),
 ("2. Deposits and Non-Payable Items", [
  "A refundable deposit of 10,000 rupees is collected for general ward admissions where cashless approval is pending. The deposit is adjusted against the final bill or refunded within 5 working days of discharge.",
  "Insurers may not cover certain items such as registration fees, toiletries, and personal comfort items. These non-payable items are listed separately in the bill and must be paid by the patient."]),
 ("3. Estimates", [
  "Before a planned surgery or procedure, patients can request a written estimate of expected charges from the billing desk. The estimate is provided within one working day. The final bill may differ if the treatment plan changes or the stay is extended."]),
 ("4. Reimbursement Claims", [
  "If you pay the bill yourself and plan to claim from your insurer, the hospital provides the required documents within 2 working days of discharge: the discharge summary, the itemised bill, prescriptions, investigation reports and payment receipts.",
  "Please check your policy for the claim submission deadline, as it varies by insurer. Keeping originals of all documents is strongly recommended."]),
 ("5. Billing Disputes", [
  "If you believe a charge is incorrect, raise a billing query with the patient accounts team within 30 days of the bill date, in person or through the hospital app. The team reviews each query and responds within 10 working days.",
  "Any amount found to be charged in error is credited back to the original mode of payment."]),
 ("6. Payment Methods", [
  "The hospital accepts cash, debit and credit cards, UPI and net banking. Payments above 2 lakh rupees must be made through bank transfer or card. A receipt is issued for every payment."]),
]),
"outpatient_medication_safety_handbook.pdf": ("CityCare Hospital - Outpatient Appointments and Medication Safety Handbook", [
 ("1. Booking and Attending Appointments", [
  "Outpatient appointments can be booked through the hospital app, the website or the appointment helpdesk. Please arrive 15 minutes before your appointment time to complete registration and basic checks.",
  "Bring your photo ID, previous prescriptions, and recent test reports. Appointments can be rescheduled free of charge up to 4 hours before the scheduled time."]),
 ("2. Teleconsultation", [
  "Video consultations are available daily from 8 AM to 8 PM for follow-up visits and non-emergency concerns. Patients need a stable internet connection and should keep their medical records ready. Teleconsultation is not suitable for emergencies or conditions that need a physical examination."]),
 ("3. Tests and Reports", [
  "Some blood tests require fasting. For a fasting blood sugar test, patients should not eat or drink anything except water for 8 to 10 hours before the sample is taken. Always follow the specific instructions given by your doctor or the lab.",
  "Routine blood test reports are available in the patient portal within 24 hours. Culture reports can take up to 48 hours or longer."]),
 ("4. Medication Safety at Home", [
  "Always carry an up-to-date list of all medicines you take, including over-the-counter medicines and supplements, and tell every doctor about any known allergies. Do not stop or change a prescribed medicine without consulting your doctor.",
  "Never share your prescription medicines with others. Store medicines as directed on the label, away from heat, moisture and the reach of children, and check expiry dates regularly."]),
 ("5. Side Effects and Pharmacy Services", [
  "If you experience an unexpected reaction or side effect after taking a medicine, contact your doctor or the hospital pharmacy. Serious reactions such as breathing difficulty, swelling of the face or severe rash need emergency care. The hospital's pharmacovigilance desk collects reports of adverse drug reactions to improve patient safety.",
  "The hospital pharmacy is open 24 hours a day. Pharmacists can explain how and when to take a medicine, but they cannot change a prescription."]),
 ("6. Emergencies", [
  "In a medical emergency such as chest pain, severe bleeding or loss of consciousness, do not wait for an appointment. Call the hospital emergency line or go to the emergency department, which operates 24 hours a day. The hospital ambulance can be requested through the emergency line."]),
]),
}
for fn,(title,secs) in DOCS.items():
    doc = SimpleDocTemplate(f"data/{fn}", pagesize=A4, title=title)
    s=[Paragraph(title,styles['Title']),Paragraph(NOTE,styles['Italic']),Spacer(1,12)]
    for i,(h,paras) in enumerate(secs):
        s.append(Paragraph(h,styles['Heading2']))
        for p in paras: s += [Paragraph(p,styles['BodyText']),Spacer(1,8)]
        if i==2: s.append(PageBreak())
    doc.build(s)
print("done")
