import pytest

from qanoon_ai.verification.quantities import extract_quantities, unsupported_quantities

THEFT = "Whoever commits theft shall be punished with imprisonment of either description for a term which may extend to three years, or with fine or with both."
INTERIM = "the Family Court may pass an interim order for maintenance, whereunder the payment shall be made by the fourteenth of each month"
RETIREMENT = "on the completion of the sixtieth year of his age"
WITNESSES = "the instrument shall be attested by two men, or one man and two women"
DOWRY = "shall not exceed five thousand rupees"


@pytest.mark.parametrize("text, expected", [
    ("may extend to three years", {(3, "year")}),
    ("up to 3 years", {(3, "year")}),
    ("teen saal tak qaid", {(3, "year")}),
    ("تین سال تک قید", {(3, "year")}),
    ("۳ سال", {(3, "year")}),
    ("twenty-four hours", {(24, "hour")}),
    ("by the fourteenth of each month", {(14, "date")}),
    ("by the 14th", {(14, "date")}),
    ("har mahine ki 14 tareekh tak", {(14, "date")}),
    ("within 14 days", {(14, "day")}),
    ("14 din mein", {(14, "day")}),
    ("sixtieth year", {(60, "year")}),
    ("Rs. 50,000", {(50000, None)}),
    ("fifty thousand rupees", {(50000, "money")}),
    ("section 379", set()),
    ("under Section 497(1) and sub-section (3)", set()),
    ("by fourteen day of each month", {(14, "date")}),
    ("5[ 6[17-A. Suit for maintenance", {(17, None)}),
])
def test_extracts_value_and_kind(text, expected):
    assert extract_quantities(text) == expected


@pytest.mark.parametrize("text", [
    "no one may do so", "ek shakhs", "wo paisay de do", "ایک شخص", "the first appearance of the defendant",
    "a second marriage", "shohar ke saath rehna",
])
def test_ordinary_words_are_not_numbers(text):
    assert extract_quantities(text) == set()


@pytest.mark.parametrize("claim, evidence", [
    ("The maximum prison term is three years.", THEFT),
    ("Chori ki saza 3 saal tak qaid ya jurmana ya dono hai.", THEFT),
    ("چوری کی سزا تین سال تک قید یا جرمانہ یا دونوں ہے۔", THEFT),
    ("Interim maintenance must be paid by the 14th of every month.", INTERIM),
    ("Har mahine ki 14 tareekh tak maintenance deni hogi.", INTERIM),
    ("A civil servant retires at 60 years of age.", RETIREMENT),
    ("It must be attested by two men or one man and two women.", WITNESSES),
    ("Dowry may not exceed Rs. 5,000.", DOWRY),
    ("Interim maintenance is payable by the 14th of each month.", "if the defendant fails to pay the maintenance by fourteen day of each month"),
    ("Under section 497(1), bail may be refused.", "he shall not be so released"),
])
def test_matching_quantities_pass(claim, evidence):
    assert unsupported_quantities(claim, evidence) == []


@pytest.mark.parametrize("claim, evidence, problem", [
    ("چوری کی سزا ایک سال تک قید ہے۔", THEFT, "1 year"),
    ("The punishment is up to seven years.", THEFT, "7 year"),
    ("Agar defendant 14 din mein maintenance nahi deta to defence strike off hota hai.", INTERIM, "14 day"),
    ("Maintenance must be paid within fourteen days.", INTERIM, "14 day"),
    ("A civil servant retires at 65.", RETIREMENT, "65"),
    ("Dowry may not exceed ten thousand rupees.", DOWRY, "10000 money"),
])
def test_changed_quantities_are_reported(claim, evidence, problem):
    assert problem in unsupported_quantities(claim, evidence)
