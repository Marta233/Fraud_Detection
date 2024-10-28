import React, { useState } from 'react';
import axios from 'axios';
import './styles.css';

const FraudForm = () => {
    const [formData, setFormData] = useState({
        purchase_time: '',
        signup_time: '',
        source: '',
        browser: '',
        sex: '',
        purchase_value: '',
        age: '',
        country: ''
    });
    
    const [result, setResult] = useState('');
    const [loading, setLoading] = useState(false);

    const handleChange = (e) => {
        setFormData({ ...formData, [e.target.name]: e.target.value });
    };

    const handleSubmit = async (e) => {
        e.preventDefault();
        setLoading(true);
        try {
            const response = await axios.post('http://127.0.0.1:5000/predict', formData);
            setResult(response.data.message);
        } catch (error) {
            console.error("There was an error!", error.response ? error.response.data : error);
            setResult("Error: Could not get prediction.");
        } finally {
            setLoading(false);
        }
    };

    return (
        <div className="background-image">
             <h2>Fraud Detection Form</h2>
            <div className="form-container">
               
                <form onSubmit={handleSubmit}>
                    <div className="form-group">
                        <label>Purchase Time</label>
                        <input
                            type="datetime-local"
                            name="purchase_time"
                            value={formData.purchase_time}
                            onChange={handleChange}
                            required
                        />
                    </div>
                    <div className="form-group">
                        <label>Signup Time</label>
                        <input
                            type="datetime-local"
                            name="signup_time"
                            value={formData.signup_time}
                            onChange={handleChange}
                            required
                        />
                    </div>
                    <div className="form-group">
                        <label>Source</label>
                        <input
                            type="text"
                            name="source"
                            value={formData.source}
                            onChange={handleChange}
                            required
                        />
                    </div>
                    <div className="form-group">
                        <label>Browser</label>
                        <input
                            type="text"
                            name="browser"
                            value={formData.browser}
                            onChange={handleChange}
                            required
                        />
                    </div>
                    <div className="form-group">
                        <label>Gender</label>
                        <input
                            type="text"
                            name="gender"
                            value={formData.gender}
                            onChange={handleChange}
                            required
                        />
                    </div>
                    <div className="form-group">
                        <label>Purchase Value</label>
                        <input
                            type="number"
                            name="purchase_value"
                            value={formData.purchase_value}
                            onChange={handleChange}
                            required
                        />
                    </div>
                    <div className="form-group">
                        <label>Age</label>
                        <input
                            type="number"
                            name="age"
                            value={formData.age}
                            onChange={handleChange}
                            required
                        />
                    </div>
                    <div className="form-group">
                        <label>Country</label>
                        <input
                            type="text"
                            name="country"
                            value={formData.country}
                            onChange={handleChange}
                            required
                        />
                    </div>
                    <button type="submit" disabled={loading}>
                        {loading ? "Loading..." : "Submit"}
                    </button>
                </form>
                {result && <h3>{result}</h3>}
            </div>
        </div>
    );
};

export default FraudForm;
